# ITAB COMPUTER SOFTWARE

## SETUP

1. Clone the repository

Open a terminal and run the following commands:
```sh
git clone https://github.com/Johannes-Pettersson/ITAB-computer-software.git
cd ITAB-computer-software
git checkout master
git pull
```

2. Create and use a Virtual Environment
To create a virtual environment, go to your project’s directory and run the following command. This will create a new virtual environment in a local folder named .venv:
```sh
python3 -m venv .venv # Unix/macOS
py -m venv .venv # Windows
```

Activate a Virtual Environment
```sh
source .venv/bin/activate # Unix/macOS
.venv\Scripts\activate # Windows
```

Install packages from requirements file
```sh
python3 -m pip install -r requirements.txt # Unix/macOS
py -m pip install -r requirements.txt # Windows
```
How to install a package inside .venv
```sh
python3 -m pip install example_package
py -m pip install example_package
```

Deactivate the environment when finished:
```sh
deactivate
```

3. Configure Environment Variables

```sh
cp .env.example .env
```

Modify .env to set the correct UART communication port:
```sh
UART_PORT=<your_desired_port>
```

4. Additional Information

For further details on usage and additional configuration options, refer to the README.md file.