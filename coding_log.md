```bash

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install requests python-dotenv

#Make extension txt
pip freeze > requirements.txt

python TUDAGPT_API.py 5
```