from src.hsrag.client import MockClient

m = MockClient()
print("Valid:", m.complete([{"role": "user", "text": "anything"}]))

broken = MockClient(broken=True)
print("Broken:", broken.complete([{"role": "user", "text": "anything"}]))