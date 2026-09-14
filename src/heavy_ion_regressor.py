# src/models/heavy_ion_regressor.py
"""
Unified loader + preprocessing + inference for the trained heavy-ion
regression models (AuAu and PbPb).

The two model families differ only in:

  * number of image channels   : 2 for AuAu (v3 preprocessing), 2 for PbPb (v4)
  * presence of pt_true_input  : yes for AuAu (evidential head), no for PbPb
  * target transform for <pT>  : log(1 + pt/max_pt) for AuAu, linear for PbPb

All three differences are inferred from the model itself.  The caller
only needs to know the directory in which the model lives.
"""

from pathlib import Path
import json
import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------
# Custom layers — required to deserialize the saved models
# ---------------------------------------------------------------------

try:
    from utils.blocks import CBAMBlock, SpatialAttention
    from utils.blocks import custom_objects as _base_co
except ImportError:
    from channelAttention import CBAMBlock, SpatialAttention
    _base_co = {}


class AddBaseline(tf.keras.layers.Layer):
    """Fixed polynomial baseline used by the AuAu <pT> head."""

    def __init__(self, coeffs, **kwargs):
        super().__init__(**kwargs)
        self.coeffs_list = [tf.constant(c, dtype=tf.float32) for c in coeffs]
        self._coeffs_cfg = list(coeffs)

    def call(self, inputs):
        sum_norm, pt_residual = inputs
        s = tf.squeeze(sum_norm, axis=-1)
        b = tf.expand_dims(tf.math.polyval(self.coeffs_list, s), -1)
        b = tf.clip_by_value(b, 0.0, 1.0)
        return pt_residual + tf.math.log1p(b)

    def get_config(self):
        cfg = super().get_config()
        cfg["coeffs"] = self._coeffs_cfg
        return cfg


class EvidentialPtHead(tf.keras.layers.Layer):
    """Pass-through head (AuAu v5).  No-op at inference."""

    def __init__(self, evidential_coeff=0.05, **kwargs):
        super().__init__(**kwargs)
        self.evidential_coeff = evidential_coeff

    def call(self, inputs, training=None):
        pt_gamma, _ev_raw, _pt_true = inputs
        return pt_gamma

    def get_config(self):
        cfg = super().get_config()
        cfg["evidential_coeff"] = self.evidential_coeff
        return cfg


class AdaptiveBaseline(tf.keras.layers.Layer):
    """Learnable baseline used by the PbPb (v7) <pT> head."""

    def __init__(self, coeffs, use_poly=False, use_nn=True, **kwargs):
        super().__init__(**kwargs)
        self.use_poly = use_poly
        self.use_nn = use_nn
        self._coeffs_list = list(coeffs)
        self.coeffs = [tf.constant(c, tf.float32) for c in coeffs]
        if self.use_nn:
            self.nn = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1),
            ])
        self.alpha = tf.Variable(0.0, trainable=True)
        self.beta = tf.Variable(1.0, trainable=True)

    def call(self, inputs):
        s, pt_res = inputs
        s_flat = tf.squeeze(s, -1)
        baseline = 0.0
        if self.use_poly:
            poly = tf.expand_dims(tf.math.polyval(self.coeffs, s_flat), -1)
            poly = tf.clip_by_value(poly, 0.0, 1.0)
            baseline = baseline + self.alpha * poly
        if self.use_nn:
            baseline = baseline + self.beta * self.nn(s)
        return pt_res + baseline

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "coeffs": self._coeffs_list,
            "use_poly": self.use_poly,
            "use_nn": self.use_nn,
        })
        return cfg


def _stub_loss(_y_true, y_pred):
    """Placeholder, only needed if a .keras was saved with compile state."""
    return tf.reduce_mean(tf.square(y_pred))


CUSTOM_OBJECTS = {
    **_base_co,
    "CBAMBlock": CBAMBlock,
    "SpatialAttention": SpatialAttention,
    "AddBaseline": AddBaseline,
    "EvidentialPtHead": EvidentialPtHead,
    "AdaptiveBaseline": AdaptiveBaseline,
    "multitask_loss": _stub_loss,
}


# ---------------------------------------------------------------------
# Preprocessor — same for both systems, controlled by the model signature
# ---------------------------------------------------------------------

class HeavyIonPreprocessor:
    """Prepare raw TRENTO output for a given trained model.

    The image transformation is identical for the two families:
        ch0 = image / 255
        ch1 = image / sum(image)
        sum_norm = sum(image) / max_sum        (from data_specs.json)

    Whether the pt target is in log space is detected from the model
    itself (presence of an AddBaseline layer), not from a hard-coded tag.
    """

    def __init__(self, specs, input_names, channels, pt_in_log_space):
        mv = specs["header"]["max_values"]
        self.max_sum = float(mv["max_sum"])
        self.max_pixel = float(mv.get("max_pixel_raw", mv.get("x_max", 255.0)))
        self.max_Nch = float(mv["max_Nch"])
        self.max_pt = float(mv["max_pt"])

        self.needs_pt_true_input = "pt_true_input" in input_names
        self.channels = channels
        self.pt_in_log_space = pt_in_log_space

    # -- forward -----------------------------------------------------

    def __call__(self, images, sums=None):
        images = np.asarray(images, dtype=np.float32)
        if images.ndim == 3:
            images = images[..., None]

        if sums is None:
            sums = images.sum(axis=(1, 2, 3))
        sums = np.asarray(sums, dtype=np.float32).reshape(-1)

        # dual-channel image construction
        if self.channels == 1:
            img = images / self.max_pixel
        elif self.channels == 2:
            ch0 = images / self.max_pixel
            ch1 = images / (sums[:, None, None, None] + 1e-8)
            img = np.concatenate([ch0, ch1], axis=-1)
        else:
            raise ValueError(f"Unsupported channel count: {self.channels}")

        sum_norm = (sums / self.max_sum).astype(np.float32).reshape(-1, 1)

        inputs = {
            "image_input": img.astype(np.float32),
            "sum_input": sum_norm,
        }
        if self.needs_pt_true_input:
            inputs["pt_true_input"] = np.zeros_like(sum_norm)

        return inputs

    # -- inverse -----------------------------------------------------

    def to_physical(self, preds):
        preds = np.asarray(preds)
        nch = preds[:, 0] * self.max_Nch
        if self.pt_in_log_space:
            pt = np.expm1(preds[:, 1]) * self.max_pt
        else:
            pt = preds[:, 1] * self.max_pt
        return nch, pt


# ---------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------

class HeavyIonRegressor:
    """Load a trained heavy-ion model and run inference end-to-end.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing the architecture JSON, weights, and
        data_specs.json.  Alternatively, pass a system tag plus type/cut
        through the ``from_tags`` classmethod.
    system : str, optional
        One of "0.2_AuAu" or "2.76_PbPb".  If not given, inferred from
        the path.
    strategy : tf.distribute.Strategy, optional
    """

    def __init__(self, model_dir, system=None, strategy=None):
        self.model_dir = Path(model_dir)
        self.system = system or self._infer_system()
        self.strategy = strategy or tf.distribute.get_strategy()
        self.model = None
        self.pre = None

    # -- path helpers ------------------------------------------------

    def _infer_system(self):
        for tag in ("0.2_AuAu", "2.76_PbPb"):
            if tag in str(self.model_dir):
                return tag
        raise ValueError(
            f"Cannot infer system from path '{self.model_dir}'. "
            f"Pass system='0.2_AuAu' or '2.76_PbPb'."
        )

    @classmethod
    def from_tags(cls, base_dir, system, type_, cut, **kwargs):
        """Load using (system, type, cut) tags.

        Example
        -------
            HeavyIonRegressor.from_tags(
                "src/trained_models",
                system="0.2_AuAu",
                type_="entropy",
                cut="pt",
            )
        """
        return cls(Path(base_dir) / system / f"{type_}_{cut}",
                   system=system, **kwargs)

    def _find_specs(self):
        for cand in (self.model_dir / "data_specs.json",
                     self.model_dir.parent / "data_specs.json"):
            if cand.exists():
                return cand
        raise FileNotFoundError(
            f"data_specs.json not found next to or above {self.model_dir}"
        )

    # -- model loading ----------------------------------------------

    def _load_model(self):
        keras_files = sorted(self.model_dir.glob("*.keras"))
        if keras_files:
            return tf.keras.models.load_model(
                str(keras_files[0]),
                custom_objects=CUSTOM_OBJECTS,
                compile=False,
            )

        arch_files = [p for p in self.model_dir.glob("*.json")
                      if p.name != "data_specs.json"]
        weight_files = sorted(self.model_dir.glob("*.h5"))
        if arch_files and weight_files:
            from keras.models import model_from_json
            with open(arch_files[0]) as f:
                model = model_from_json(f.read(), custom_objects=CUSTOM_OBJECTS)
            model.load_weights(str(weight_files[0]))
            return model

        raise FileNotFoundError(
            f"No loadable model in {self.model_dir}. Expected one of:\n"
            f"  *.keras\n"
            f"  *.architecture.json + *.weights.h5"
        )

    # -- variant detection ------------------------------------------

    @staticmethod
    def _pt_in_log_space(model):
        """AuAu (v5) applies log1p; PbPb (v7) does not."""
        for layer in model.layers:
            if type(layer).__name__ == "AddBaseline":
                return True
        return False

    # -- public API --------------------------------------------------

    def load(self):
        with open(self._find_specs()) as f:
            specs = json.load(f)

        self.model = self._load_model()
        input_names = list(self.model.input_names)
        channels = int(self.model.inputs[0].shape[-1])
        pt_log = self._pt_in_log_space(self.model)

        self.pre = HeavyIonPreprocessor(specs, input_names, channels, pt_log)

        print(
            f"[HeavyIonRegressor] {self.system}  "
            f"inputs={input_names}  channels={channels}  "
            f"pt_log_space={pt_log}"
        )
        return self

    def predict(self, images, sums=None, batch_size=256):
        """Run inference on raw TRENTO events.

        Parameters
        ----------
        images : (N, 280, 280) or (N, 280, 280, 1) raw counts.  For
            the entropy configurations the input must already be the
            entropy density (the EoS conversion is done upstream).
        sums : (N,), optional.  Raw total per event.  If None, computed
            from the image.  Must be in the SAME units as the training
            total (energy density or entropy density, respectively).
        batch_size : int

        Returns
        -------
        nch : (N,) physical <N_ch>
        pt  : (N,) physical <p_T>, in the same units as max_pt
        """
        if self.model is None or self.pre is None:
            raise RuntimeError("Call .load() before .predict()")

        inputs = self.pre(images, sums)
        n = inputs["image_input"].shape[0]

        nch_list, pt_list = [], []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = {k: v[start:end] for k, v in inputs.items()}
            out = self.model(batch, training=False).numpy()
            nch_b, pt_b = self.pre.to_physical(out)
            nch_list.append(nch_b)
            pt_list.append(pt_b)

        return np.concatenate(nch_list), np.concatenate(pt_list)