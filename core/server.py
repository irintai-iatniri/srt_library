import asyncio
import json
import threading
import time

import numpy as np
import websockets

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
    # Support both PyTorch (detach().cpu()) and Pure Syntonic (to_list())
    flat_weights = []

    # Allow models that expose either `parameters()` (torch-like) OR
    # `get_weights()` (pure syntonic models returning ResonantTensors).
    sources = None
    if hasattr(model, "parameters"):
        sources = model.parameters()
    elif hasattr(model, "get_weights"):
        sources = model.get_weights()
    else:
        # Nothing we can introspect
        return

    for p in sources:
        w_data = None

        # Handle Pure Syntonic Parameter objects / ResonantTensor
        if hasattr(p, "to_list"):
            try:
                w_data = np.array(p.to_list())
            except Exception:
                pass

        # Handle objects with `.tensor.to_floats()` access
        if w_data is None and hasattr(p, "tensor") and hasattr(p.tensor, "to_floats"):
            try:
                w_data = np.array(p.tensor.to_floats())
            except Exception:
                pass

        # Handle PyTorch Tensor-like objects
        if w_data is None and hasattr(p, "detach"):
            try:
                w_data = p.detach().cpu().numpy().flatten()
            except Exception:
                pass

        if w_data is not None and w_data.size > 0:
            flat_weights.append(w_data)

        # Limit the size to avoid heavy serialization
        current_len = sum(len(w) for w in flat_weights)
        if current_len > 6144:  # 2048 * 3
            break

    if not flat_weights:
        return

    full_vector = np.concatenate(flat_weights)
    if len(full_vector) > 6144:
        full_vector = full_vector[:6144]  # Limit to 2048 * 3

    # Normalize for visualization (-50 to 50 range)
    if full_vector.std() > 0:
        full_vector = (
            (full_vector - full_vector.mean()) / (full_vector.std() + 1e-6) * 25.0
        )

    # Debug log to help verify monitor updates server-side
    try:
        print(
            f">>> VIZ: pushing {len(full_vector)} weights, syntony={syntony:.4f}, temp={temp:.4f}, phase={phase}"
        )
    except Exception:
        pass

    LATEST_STATE = {
        "weights": full_vector.tolist(),
        "syntony": float(syntony),
        "temperature": float(temp),
        "phase": "D" if phase > 0.5 else "H",
        "seq": LATEST_STATE.get("seq", 0) + 1,
        "ts": time.time(),
    }
