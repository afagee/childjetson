"""
Siren control — bật/tắt relay theo trạng thái (xe trong zone = bật, ra zone = tắt).
"""
import config
import state
import relay


def turn_on():
    """Bật còi (relay)."""
    with state.siren_lock:
        if state.siren_is_on:
            return
        relay.init_pins(config.SIREN_GPIO_CHANNEL, config.RELAY_CH2_GPIO)
        relay.set_channel(0, True)
        state.siren_is_on = True
        print("[JETSON] Siren ON")


def turn_off():
    """Tắt còi (relay)."""
    with state.siren_lock:
        if not state.siren_is_on:
            return
        relay.set_channel(0, False)
        state.siren_is_on = False
        print("[JETSON] Siren OFF")
