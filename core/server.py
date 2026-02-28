import asyncio
import json
import math
import threading
import time

try:
    import websockets
except ImportError:
    websockets = None

# Global state buffer (Thread-safe)
LATEST_STATE = {
    "weights": [],
    "syntony": 0.0,
    "temperature": 0.0,
    "phase": "D",
    "seq": 0,
    "ts": 0.0,
}
# Control state writable by websocket clients
CONTROL_STATE = {
    "phase": "D",
    "temperature": 0.0,
    "commands": [],  # recent commands for diagnostics
}


async def handler(websocket):
    """Push state to the browser at 60fps"""
    while True:
        try:
            # Try to receive an incoming message (non-blocking poll)
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                try:
                    cmd = json.loads(msg)
                    # Simple command handling
                    if isinstance(cmd, dict):
                        if "cmd" in cmd:
                            if cmd["cmd"] == "set_phase" and "phase" in cmd:
                                CONTROL_STATE["phase"] = cmd["phase"]
                                CONTROL_STATE["commands"].append(cmd)
                            if cmd["cmd"] == "set_temperature" and "temperature" in cmd:
                                CONTROL_STATE["temperature"] = float(cmd["temperature"])
                                CONTROL_STATE["commands"].append(cmd)
                except Exception:
                    pass
            except asyncio.TimeoutError:
                # no incoming message this frame
                pass

            # Serialize the latest state and include current control state
            payload = dict(LATEST_STATE)
            payload["control"] = {
                "phase": CONTROL_STATE.get("phase", "D"),
                "temperature": CONTROL_STATE.get("temperature", 0.0),
            }
            data = json.dumps(payload)
            await websocket.send(data)
            await asyncio.sleep(0.016)  # ~60 FPS
        except websockets.exceptions.ConnectionClosed:
            break


async def start_server():
    print(">>> SYNTONIC CONSOLE: Listening on localhost:8765")
    async with websockets.serve(handler, "localhost", 8765):
        # Create a Future that never completes to keep the server running
        await asyncio.Future()


def launch_background_thread():
    """Call this from trainer.py to start the server non-blocking"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    t = threading.Thread(
        target=loop.run_until_complete, args=(start_server(),), daemon=True
    )
    t.start()


def update_monitor(model, syntony, temp, phase):
    """Called by the Trainer loop to update global state"""
    global LATEST_STATE

    # Extract raw weights from the model (CPU copy)
    flat_weights = []

    sources = None
    if hasattr(model, "parameters"):
        sources = model.parameters()
    elif hasattr(model, "get_weights"):
        sources = model.get_weights()
    else:
        return

    for p in sources:
        w_data = None

        # Handle Pure Syntonic Parameter objects / ResonantTensor
        if hasattr(p, "to_list"):
            try:
                w_data = list(p.to_list())
            except Exception:
                pass

        if w_data is None and hasattr(p, "tensor") and hasattr(p.tensor, "to_floats"):
            try:
                w_data = list(p.tensor.to_floats())
            except Exception:
                pass

        if w_data is not None and len(w_data) > 0:
            flat_weights.extend(w_data)

        if len(flat_weights) > 6144:
            break

    if not flat_weights:
        return

    full_vector = flat_weights[:6144]

    # Normalize for visualization (-50 to 50 range)
    n = len(full_vector)
    if n > 0:
        mean = sum(full_vector) / n
        var = sum((v - mean) ** 2 for v in full_vector) / n
        std = math.sqrt(var) if var > 0 else 0.0
        if std > 0:
            full_vector = [(v - mean) / (std + 1e-6) * 25.0 for v in full_vector]

    try:
        print(
            f">>> VIZ: pushing {len(full_vector)} weights, syntony={syntony:.4f}, temp={temp:.4f}, phase={phase}"
        )
    except Exception:
        pass

    LATEST_STATE = {
        "weights": full_vector,
        "syntony": float(syntony),
        "temperature": float(temp),
        "phase": "D" if phase > 0.5 else "H",
        "seq": LATEST_STATE.get("seq", 0) + 1,
        "ts": time.time(),
    }
