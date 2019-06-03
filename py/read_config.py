import configparser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("ConfigFile", help="Config file")
args = parser.parse_args()

config = configparser.ConfigParser(delimiters=' ')

f = open(args.ConfigFile, 'r')
ctext = "[CONFIG]\n" + f.read()
f.close()

config.read_string(ctext)
config_parameters = config['CONFIG']

print(config_parameters['FileNameRList'])
# for key in config_parameters:
#     print(key, config_parameters[key])
