from jtop import jtop, JtopException
import csv
import argparse
import subprocess

def del_stats(stats):
    del stats['jetson_clocks']
    del stats['nvp model']
    del stats['uptime']
    del stats['NVENC']
    del stats['NVDEC']
    del stats['NVJPG']
    del stats['fan']
    del stats["Temp AO"]            
    del stats["Temp CPU"]            
    del stats["Temp GPU"]
    del stats["Temp PLL"]
    del stats["Temp thermal"]

    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple jtop logger')
    # Standard file to store the logs
    parser.add_argument('--file', action="store", dest="file", default="log.csv")
    parser.add_argument('--run', type=str, help="Command to be executed.")
    args = parser.parse_args()

    print("Simple jtop logger")
    print("Saving log on {file}".format(file=args.file))

    try:
        with jtop() as jetson:
            # Make csv file and setup csv
            with open(args.file, 'w') as csvfile:
                stats = jetson.stats
                # Initialize cws writer
                stats = del_stats(stats)
                writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
                # Write header
                writer.writeheader()
                # Write first row
                
                writer.writerow(stats)
                # Start loop
                print("Trying to run " + args.run)
                process = subprocess.Popen(args.run.split(" "))
                # process2 = subprocess.Popen(args.run.split(" "))
                # TODO run external program
                while jetson.ok():
                    stats = jetson.stats
                    stats = del_stats(stats)
                    # Write row
                    writer.writerow(stats)
                    # print("Log at {time}".format(time=stats['time']))
                    if process.poll() is not None:# and process2.poll() is not None:
                        break
    except JtopException as e:
        print(e)
    except KeyboardInterrupt:
        print("Closed with CTRL-C")
    except IOError:
        print("I/O error")