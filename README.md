# Iperf2 Extract

Script for Iperf 2 Extract, Transform and Load (ETL).

## Requirements

- Python >=3.8
- Log trace of Iperf 2 using the `-o` option
- Tested with:
  - iperf version 2.1.n (24 April 2023) pthreads

## Installation

Install requirements with: 

`pip3.10 install -r requirements.txt`

## Usage

### General command

```
usage: iperf2-etl.py [-h] {serialize,plot} ...

Parse, save and plot Iperf2 trace

positional arguments:
  {serialize,plot}
    serialize       Convert iperf2 log file to json
    plot            Plot the iperf results

options:
  -h, --help        show this help message and exit
```

### Plot command

```
usage: iperf2-etl.py serialize [-h] -c <client trace> -s <server trace> -o <name>

options:
  -h, --help            show this help message and exit
  -c <client trace>, --client <client trace>
                        Specify the client log file to parse
  -s <server trace>, --server <server trace>
                        Specify the server log file to parse
  -o <name>, --output <name>
                        Set the output folder
```

### Serialize command

```
usage: iperf2-etl.py plot [-h] -c <client trace> -s <server trace> -o <name> [--latex]

options:
  -h, --help            show this help message and exit
  -c <client trace>, --client <client trace>
                        Specify the client log file to parse
  -s <server trace>, --server <server trace>
                        Specify the server log file to parse
  -o <name>, --output <name>
                        Set the output folder
  --latex               Generate tikz file for Latex integration using Tikz
```

## How to plot

Save the client and the server log files. Example can be found in the `example` folder.

Run `python3.10 iperf2-etl.py plot -c example/client_1.log -s example/server_1.log -o client_1 --latex`

The `--latex` flag can be omited if you don't want to have the **tikz** files for LaTeX reports or publications.

You can find and example of the extracted file in the **client_1** folder