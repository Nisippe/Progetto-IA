import os

def generate_route_file(n):
    print(os.system('cmd /c "dir & cd XML & py "C:/Program Files (x86)/Eclipse/Sumo/tools/randomTrips.py" -n TEST1.net.xml -e {}"'.format(n)))


if __name__ == "__main__":
    generate_route_file(10)