"""
MIT License

Copyright (c) 2023 Youssouf Drif

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import matplotlib.pyplot as plt
import tikzplotlib
import sys
import logging
import os
import codecs
import re
import json
import numpy as np

from typing import Final, Dict


RE_TCP: Final[str] = r"\W*(TCP)\W*"
RE_REVERSE: Final[str] = r"\W*(reverse)\W*"
RE_TRIP_TIMES: Final[str] = r"\W*(trip-times)\W*"
RE_VAL_COMPLETE: Final[
    str] = r"\[ \*[0-9]+\] [0-9]*[.][0-9]*\-[0-9]*[.][0-9]* sec *[0-9]*[.][0-9]* [a-zA-Z]* *([0-9]*[.][0-9]*) ([a-zA-Z]*)/sec *([0-9]*[.][0-9]*) ([a-z]*) *[0-9\/]* *\(([0-9]*[.]?[0-9]+)\%\) ([+-]?[0-9]*[.][0-9]*).*"

PROTO_UDP: Final[int] = 0x01
PROTO_TCP: Final[int] = 0x02


class IperfOption(object):

    def __init__(self) -> None:
        self.protocol: int = None
        self.trip_times: bool = None
        self.reverse: bool = None
        self.output_folder: str = None

        self.client_raw: str = None
        self.server_raw: str = None
        self.raw: str = None

        self.regex: str = None

    def to_dict(self) -> None:
        j = {
            'protocol': self.protocol,
            'trip_times': self.trip_times,
            'reverse': self.reverse,
        }
        return j

    @classmethod
    def build_options(cls, client, server, output) -> object:
        options = IperfOption()
        options.output_folder = output
        with codecs.open(client, 'r', 'utf8') as f:
            options.client_raw = f.read()
        with codecs.open(server, 'r', 'utf8') as f:
            options.server_raw = f.read()

        options.protocol = PROTO_TCP if len(re.findall(
            RE_TCP, options.client_raw, re.MULTILINE)) > 0 else PROTO_UDP
        protocol_name = "TCP" if options.protocol == PROTO_TCP else "UDP"
        logging.info(f"Protocol {protocol_name} detected")

        options.reverse = len(re.findall(
            RE_REVERSE, options.client_raw, re.MULTILINE)) > 0
        logging.info(f"Option reverse: {options.reverse}")

        options.trip_times = len(re.findall(
            RE_TRIP_TIMES, options.client_raw, re.MULTILINE)) > 0
        logging.info(f"Option trip-times: {options.trip_times}")

        if options.reverse:
            options.raw = options.client_raw
            if options.trip_times:
                options.regex = RE_VAL_COMPLETE

        return options


class Iperf(object):

    def __init__(self) -> None:
        self.options: IperfOption = None

        self.bandwidth: np.array = np.array([], dtype=float)
        self.jitter: np.array = np.array([], dtype=float)
        self.per: np.array = np.array([], dtype=float)
        self.latency: np.array = np.array([], dtype=float)

        self.unit_time: str = "seconds"
        self.unit_bw: str = None
        self.unit_latency: str = None
        self.unit_jitter: str = None

        self.sample_number: int = None

    def save_latex(self) -> None:
        if not os.path.exists(f"{self.options.output_folder}/tikz"):
            os.makedirs(f"{self.options.output_folder}/tikz")
        plt.clf()
        plt.cla()
        plt.close()

        plt.figure()
        columns = [self.bandwidth, self.jitter, self.latency, self.per]
        plt.boxplot(columns,
                    positions=[2, 4, 6, 8],
                    widths=1.5,
                    patch_artist=True,
                    showfliers=False,
                    showmeans=True,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                              "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5})
        plt.xticks([2, 4, 6, 8], ["Bandwidth", "Jitter",
                   "Latency", "Packet Error Rate"])
        plt.title(f"Box plot")

        # tikzplotlib.clean_figure()
        tikzplotlib.save(f"{self.options.output_folder}/tikz/boxplot.tikz",
                         textsize=5, axis_width="\\textwidth")

        plt.figure()
        plt.plot(np.arange(self.sample_number),
                 self.bandwidth, label="Throughput")
        plt.xlabel("Seconds")
        plt.ylabel(self.unit_bw)
        plt.title("Throughput")
        tikzplotlib.clean_figure()
        tikzplotlib.save(f"{self.options.output_folder}/tikz/throughput.tikz",
                         textsize=5, axis_width="\\textwidth")

        plt.figure()
        plt.plot(np.arange(self.sample_number), self.jitter, label="Jitter")
        plt.xlabel("Seconds")
        plt.ylabel(self.unit_jitter)
        plt.title("Jitter")
        tikzplotlib.clean_figure()
        tikzplotlib.save(f"{self.options.output_folder}/tikz/jitter.tikz",
                         textsize=5, axis_width="\\textwidth")

        plt.figure()
        plt.plot(np.arange(self.sample_number),
                 self.per, label="Packet Error Rate")
        plt.xlabel("Seconds")
        plt.ylabel("Percentage (%)")
        plt.title("Packet Error Rate")
        tikzplotlib.clean_figure()
        tikzplotlib.save(f"{self.options.output_folder}/tikz/packet_error_rate.tikz",
                         textsize=5, axis_width="\\textwidth")

        if self.options.trip_times:
            plt.figure()
            plt.plot(np.arange(self.sample_number),
                     self.latency, label="Trip time")
            plt.xlabel("Seconds")
            plt.ylabel(self.unit_latency)
            plt.title("Trip time")
            tikzplotlib.clean_figure()
            tikzplotlib.save(f"{self.options.output_folder}/tikz/trip_time.tikz",
                             textsize=5, axis_width="\\textwidth")


    def plot(self) -> None:
        if not os.path.exists(f"{self.options.output_folder}/images"):
            os.makedirs(f"{self.options.output_folder}/images")
        plt.clf()
        plt.cla()
        plt.close()

        plt.figure()
        columns = [self.bandwidth, self.jitter, self.latency, self.per]
        plt.boxplot(columns,
                    positions=[2, 4, 6, 8],
                    widths=1.5,
                    patch_artist=True,
                    showfliers=False,
                    showmeans=True,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                              "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5})
        plt.xticks([2, 4, 6, 8], ["Bandwidth", "Jitter",
                   "Latency", "Packet Error Rate"])
        plt.title(f"Box plot")
        plt.savefig(fname=f"{self.options.output_folder}/images/boxplot.pdf")

        plt.figure()
        plt.plot(np.arange(self.sample_number),
                 self.bandwidth, label="Throughput")
        plt.xlabel("Seconds")
        plt.ylabel(self.unit_bw)
        plt.title("Throughput")
        plt.savefig(
            fname=f"{self.options.output_folder}/images/throughput.pdf")

        plt.figure()
        plt.plot(np.arange(self.sample_number), self.jitter, label="Jitter")
        plt.xlabel("Seconds")
        plt.ylabel(self.unit_jitter)
        plt.title("Jitter")
        plt.savefig(fname=f"{self.options.output_folder}/images/jitter.pdf")

        plt.figure()
        plt.plot(np.arange(self.sample_number),
                 self.per, label="Packet Error Rate")
        plt.xlabel("Seconds")
        plt.ylabel("Percentage (%)")
        plt.title("Packet Error Rate")
        plt.savefig(
            fname=f"{self.options.output_folder}/images/packet_error_rate.pdf")

        if self.options.trip_times:
            plt.figure()
            plt.plot(np.arange(self.sample_number),
                     self.latency, label="Trip time")
            plt.xlabel("Seconds")
            plt.ylabel(self.unit_latency)
            plt.title("Trip time")
            plt.savefig(
                fname=f"{self.options.output_folder}/images/trip_time.pdf")

        plt.show()

    def to_dict(self) -> Dict[str, str]:
        j = {
            'bandwidth': self.bandwidth.tolist(),
            'jitter': self.jitter.tolist(),
            'latency': self.latency.tolist(),
            'per': self.per.tolist(),
            'options': self.options.to_dict(),
        }
        return j

    def to_json(self) -> None:
        j = self.to_dict()
        if not os.path.exists(f"{self.options.output_folder}/json"):
            os.makedirs(f"{self.options.output_folder}/json")
        filepath = f"{self.options.output_folder}/json/etl.json"
        json.dump(j, codecs.open(filepath, 'w', encoding='utf-8'),
                  separators=(',', ':'),
                  sort_keys=True,
                  indent=4,
                  )
        logging.info(f"Extracted data in JSON file: {filepath}")

    @classmethod
    def from_file(cls, options: IperfOption):
        iperf = Iperf()
        iperf.options = options
        values = re.findall(options.regex, options.raw, re.MULTILINE)
        if len(values) == 0:
            logging.error("No values to parse, exiting...")
            sys.exit(1)

        logging.info("Extracting data from trace file")
        bw = []
        jitter = []
        latency = []
        per = []

        iperf.unit_bw = values[0][1]
        iperf.unit_latency = values[0][3]
        iperf.unit_jitter = values[0][3]

        iperf.sample_number = len(values[:-1])

        logging.info(f"Number of samples: {iperf.sample_number}")
        for v in values[:-1]:
            bw.append(float(v[0]))
            jitter.append(float(v[2]))
            per.append(float(v[4]))
            latency.append(abs(float(v[5])))

        logging.info("Extracting bandwidth values")
        iperf.bandwidth = np.array(bw, dtype=float)
        logging.info("Extracting jitter values")
        iperf.jitter = np.array(jitter, dtype=float)
        logging.info("Extracting latency values")
        iperf.latency = np.array(latency, dtype=float)
        logging.info("Extracting packet error rate values")
        iperf.per = np.array(per, dtype=float)
        return iperf


if __name__ == "__main__":

    logging.basicConfig(
        format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', encoding="utf-8", level=logging.INFO)

    # Define global parser and parse arguments
    parser = argparse.ArgumentParser(
        description="Parse, save and plot Iperf2 trace")

    subparsers = parser.add_subparsers(dest="subparser")

    # Define serizlizer subparser
    serialize_parser = subparsers.add_parser(
        "serialize", help="Convert iperf2 log file to json")
    serialize_parser.add_argument("-c", "--client", help="Specify the client log file to parse",
                                  type=str, default=None, metavar=('<client trace>'), required=True)
    serialize_parser.add_argument("-s", "--server", help="Specify the server log file to parse",
                                  type=str, default=None, metavar=('<server trace>'), required=True)
    serialize_parser.add_argument("-o", "--output", help="Set the output folder",
                                  type=str, default=None, metavar=('<name>'), required=True)

    # Define plot subparser
    plot_parser = subparsers.add_parser("plot", help="Plot the iperf results")

    plot_parser.add_argument("-c", "--client", help="Specify the client log file to parse",
                             type=str, default=None, metavar=('<client trace>'), required=True)
    plot_parser.add_argument("-s", "--server", help="Specify the server log file to parse",
                             type=str, default=None, metavar=('<server trace>'), required=True)
    plot_parser.add_argument("-o", "--output", help="Set the output folder",
                             type=str, default=None, metavar=('<name>'), required=True)
    plot_parser.add_argument("--latex", action="store_true", default=False,
                             help="Generate tikz file for Latex integration using Tikz")

    args = parser.parse_args()

    if args.subparser == "serialize":
        client_log = os.path.abspath(args.client)
        if not os.path.isfile(client_log):
            logging.error(f"No file {client_log} found, exiting...")
            sys.exit(1)

        options: IperfOption = IperfOption.build_options(
            args.client, args.server, args.output)

        iperf = Iperf.from_file(options)
        iperf.to_json()

    elif args.subparser == "plot":
        client_log = os.path.abspath(args.client)
        if not os.path.isfile(client_log):
            logging.error(f"No file {client_log} found, exiting...")
            sys.exit(1)

        options: IperfOption = IperfOption.build_options(
            args.client, args.server, args.output)
        iperf = Iperf.from_file(options)
        if args.latex:
            iperf.save_latex()
        iperf.plot()
