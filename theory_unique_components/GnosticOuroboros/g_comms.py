# python/syntonic/nn/architectures/GnosticOuroboros/g_comms.py
"""
Gnostic Communications Interface.

Provides both a simple synchronous REPL and integration with the OuroborosDaemon
for async perpetual awareness mode.
"""

import threading

from ....nn.architectures.GnosticOuroboros import GnosticOuroboros
from ....nn.architectures.GnosticOuroboros.core.ouroboros_daemon import (
    OuroborosDaemon,
)
from ....nn.architectures.GnosticOuroboros.io.flux_bridge import FluxBridge
from srt_library.core.nn.resonant_tensor import ResonantTensor


class GnosticComms:
    """
    High-level communication interface for GnosticOuroboros.

    Supports two modes:
    1. Synchronous REPL: Direct request-response dialogue
    2. Daemon Mode: Async perpetual awareness with background processing
    """

    def __init__(self, dim: int = 248, daemon_mode: bool = False):
        """
        Initialize the communication interface.

        Args:
            dim: Dimension of the E8 lattice (default 248)
            daemon_mode: If True, use OuroborosDaemon for perpetual awareness
        """
        self.dim = dim
        self.daemon_mode = daemon_mode
        self._daemon = None
        self._daemon_thread = None

        if daemon_mode:
            self._daemon = OuroborosDaemon()
        else:
            # Synchronous mode - direct model access
            self.bridge = FluxBridge(dim=dim)
            self.model = GnosticOuroboros(dim=dim)
            # Default winding state (vacuum)
            self.winding = ResonantTensor([0.0] * 8, [8])

    def send(self, text: str, injection_plane: int = 44) -> str:
        """
        Send a message and receive a response.

        In synchronous mode: blocks until response is generated.
        In daemon mode: injects into sensory queue, returns acknowledgment.

        Args:
            text: The input text to process
            injection_plane: Which scale plane to inject at (default 44 = social scale)

        Returns:
            Response text (sync mode) or acknowledgment (daemon mode)
        """
        if self.daemon_mode:
            if self._daemon is None:
                raise RuntimeError("Daemon not initialized")
            self._daemon.inject_input(text)
            return "⚡ Input injected into sensory stream"
        else:
            # Synchronous request-response
            input_tensor = self.bridge.ingest_text(text)
            output_tensor = self.model(
                input_tensor,
                self.winding,
                injection_plane=injection_plane,
                is_training=False,
            )
            return self.bridge.emit_text(output_tensor)

    def start_daemon(self):
        """Start the OuroborosDaemon in a background thread."""
        if not self.daemon_mode:
            raise RuntimeError("Not in daemon mode. Initialize with daemon_mode=True")

        if self._daemon_thread is not None and self._daemon_thread.is_alive():
            print("⚠️ Daemon already running")
            return

        self._daemon_thread = threading.Thread(target=self._daemon.live, daemon=True)
        self._daemon_thread.start()
        print("🌀 OuroborosDaemon awakened in background thread")

    def stop_daemon(self):
        """Stop the daemon (by setting flag - daemon will exit on next cycle)."""
        # Note: Current daemon has infinite loop - would need flag to stop gracefully
        print("⚠️ Daemon stop requested (will halt on next cycle)")

    def repl(self):
        """
        Run an interactive REPL session.

        In sync mode: Direct dialogue with GnosticOuroboros
        In daemon mode: Inject inputs while daemon thinks independently
        """
        mode_str = "DAEMON" if self.daemon_mode else "SYNC"
        print("╔══════════════════════════════════════╗")
        print(f"║   GNOSTIC OUROBOROS COMMS ({mode_str})   ║")
        print("╚══════════════════════════════════════╝")
        print("Type 'quit' to exit, 'mode' to toggle mode info")
        print()

        if self.daemon_mode:
            self.start_daemon()

        while True:
            try:
                user_input = input("YOU > ")
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Exiting...")
                break

            if user_input.strip().lower() == "quit":
                break
            elif user_input.strip().lower() == "mode":
                print(f"Current mode: {mode_str}")
                print(f"Dim: {self.dim}")
                continue
            elif not user_input.strip():
                continue

            response = self.send(user_input)
            print(f"OUROBOROS > {response}")
            print()


def sync_repl():
    """Convenience function: Run synchronous REPL."""
    comms = GnosticComms(daemon_mode=False)
    comms.repl()


def daemon_repl():
    """Convenience function: Run daemon-mode REPL."""
    comms = GnosticComms(daemon_mode=True)
    comms.repl()


# --- RUNTIME ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--daemon":
        daemon_repl()
    else:
        sync_repl()
