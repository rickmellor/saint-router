import subprocess
import sys


def test_cli_help_lists_commands():
    out = subprocess.run([sys.executable, "-m", "goorouter", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert "serve" in out.stdout


def test_cli_serve_help():
    out = subprocess.run([sys.executable, "-m", "goorouter", "serve", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    out_low = out.stdout.lower()
    assert "host" in out_low or "port" in out_low
