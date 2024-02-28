import time
from fishers import SonarFisher, SimpleFisher

start_delay = 4

fisher = SimpleFisher()

print(f"wait {start_delay} seconds")
time.sleep(start_delay)

fisher.start()