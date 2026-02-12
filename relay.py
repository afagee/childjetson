"""
Relay GPIO for Jetson Nano — 2 channels, siren on channel 1.
Uses Jetson.GPIO (BCM). No-op or log when not on Jetson.
"""
import threading
import time

_relay_lock = threading.Lock()
_initialized = False
_use_gpio = False
_gpio = None

# Pin numbers (BCM)
SIREN_GPIO = None
RELAY_CH2_GPIO = None


def init_pins(siren_gpio, relay_ch2_gpio):
    """Set GPIO pin numbers and optionally init hardware."""
    global SIREN_GPIO, RELAY_CH2_GPIO, _initialized, _use_gpio, _gpio
    SIREN_GPIO = siren_gpio
    RELAY_CH2_GPIO = relay_ch2_gpio
    if _initialized:
        return
    try:
        import Jetson.GPIO as GPIO
        _gpio = GPIO
        _gpio.setmode(_gpio.BCM)
        _gpio.setwarnings(False)
        _gpio.setup(SIREN_GPIO, _gpio.OUT, initial=_gpio.LOW)
        _gpio.setup(RELAY_CH2_GPIO, _gpio.OUT, initial=_gpio.LOW)
        _use_gpio = True
        _initialized = True
        print("[RELAY] GPIO initialized (Jetson)")
    except Exception as e:
        _use_gpio = False
        _initialized = True
        print(f"[RELAY] Not on Jetson or GPIO unavailable: {e}")


def set_channel(channel_id, on):
    """Set relay channel to on (True) or off (False). channel_id: 0 = siren, 1 = ch2."""
    with _relay_lock:
        if not _use_gpio or _gpio is None:
            return
        pin = SIREN_GPIO if channel_id == 0 else RELAY_CH2_GPIO
        _gpio.output(pin, _gpio.HIGH if on else _gpio.LOW)


def toggle_siren(duration_sec):
    """Turn siren channel on for duration_sec then off."""
    set_channel(0, True)
    try:
        time.sleep(duration_sec)
    finally:
        set_channel(0, False)
