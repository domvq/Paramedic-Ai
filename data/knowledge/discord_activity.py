from pypresence import Presence
import time

client_id = "1544160183373267034" "

RPC = Presence(client_id)
RPC.connect()

RPC.update(
    state="Writing Python code",
    details="Working on my project"
)

print("Discord Rich Presence is running!")

while True:
    time.sleep(15)