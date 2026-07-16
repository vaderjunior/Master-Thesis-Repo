```bash

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install requests python-dotenv

#Make extension txt
pip freeze > requirements.txt

python -m scripts.test_client    # from root(self)
Get-Content experiments\results\api_log.jsonl

#versions
>> python -m scripts.test_client
>> 
>> # fallback off
>> python -m scripts.test_client --no-fallback
>> 
>> # test a different tier
>> python -m scripts.test_client --tier fast
>> 
>> # combine
>> python -m scripts.test_client --tier medium --no-fallback
```