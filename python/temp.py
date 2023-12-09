import os
array=[1,5,10]
def plot():
    for j in range(3):
        for i in array:
            print(i)
            os.system('cmd /c "py "C:/Users/drugo/Desktop/PROGETTO-IA/Python/plot.py" -f "C:/Users/drugo/Desktop/PROGETTO-IA/outputs/Q-Learning/grid_run{}_conn{}_ep{}.csv""'
                      .format(j+1,j,i))
            
if __name__ == "__main__":
    plot()