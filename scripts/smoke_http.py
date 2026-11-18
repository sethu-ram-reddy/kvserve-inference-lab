"""Exercise the actual loopback HTTP server, concurrent clients and disconnect cleanup."""
import asyncio
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import httpx


async def exercise(base):
    async with httpx.AsyncClient(base_url=base, timeout=10, trust_env=False) as client:
        results = await asyncio.gather(*[client.post('/v1/completions', json={
            'prompt': f'cache request {i}', 'max_tokens': 24, 'seed': i}) for i in range(8)])
        assert all(r.status_code == 200 and r.json()['usage']['completion_tokens'] == 24 for r in results)
        async with client.stream('POST', '/v1/completions', json={
            'prompt': 'cancel after first event', 'max_tokens': 240, 'stream': True}) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    assert json.loads(line[6:])['event'] == 'token'
                    break
        deadline = time.monotonic() + 5
        while True:
            metrics = (await client.get('/metrics')).json()
            if metrics['running'] == 0 and metrics['waiting'] == 0:
                break
            if time.monotonic() > deadline:
                raise RuntimeError('Disconnect left pending inference work')
            await asyncio.sleep(.01)
        assert metrics['assigned_kv_slots'] == metrics['reserved_kv_slots'] == 0
        report = {'concurrent_completions': 8, 'tokens_per_completion': 24,
                  'stream_first_token_observed': True, 'early_stream_close': True,
                  'no_pending_work_after_close': True, 'kv_released': True,
                  'server_counters': metrics,
                  'scope': 'loopback integration smoke test; no transport performance claim'}
        # Uptime is incidental and is not a benchmark measurement.
        report['server_counters'].pop('uptime_s')
        return report


def main():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'kvserve.api:app',
                                '--host', '127.0.0.1', '--port', str(port), '--log-level', 'error'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        base = f'http://127.0.0.1:{port}'
        deadline = time.monotonic() + 10
        while True:
            try:
                if httpx.get(base + '/health', trust_env=False).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline or process.poll() is not None:
                raise RuntimeError('Server did not start')
            time.sleep(.05)
        report = asyncio.run(exercise(base))
        Path('results/reference/http-smoke.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        process.terminate()
        process.communicate(timeout=5)


if __name__ == '__main__':
    main()
