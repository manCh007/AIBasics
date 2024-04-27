## How to start

>> Create a folder
>> Open the folder in command prompt
>> Guessing you have the python installed from windows store run the following command to instll pipx

python3 -m pip install --user pipx

>> Run the following command to check the pipx version 

pipx --version

>> If you get this error 'WARNING: The script pipx.exe is installed in...' then go to the file location mentioned and run the following command in command prompt,

.\pipx.exe ensurepath

>> Now run the following command to the inital command prompt

pipx install poetry

>> once done heck the poetry version 

poetry --version

>> Now initialize the poetry in the folder created 

poetry init

>> It will ask for a lot of details provide all and a pyproject.toml will be created this will be the blueprint of the virtual env. 
>> Run the following command to set the poetry env path to my local path,

poetry config virtualenvs.in-project true

>> Now run the following command to run install the virtual environment

poetry install

>> This will create a .venv folder inside the folder holding all binary and libraries. 
>> Now to run any python file (main.py) you can run 

poetry run python main.py

>> To add any dependency you need to run the following command 

poetry add requests 

>> This will add the dependency in the pyproject.toml file as well as in the poetry.lock file.