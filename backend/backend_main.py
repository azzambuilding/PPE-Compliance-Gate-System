#commiting table changes script
import connection_setup
import os


try:
    cur = conn.cursor()
    conn.commit()
    print(f"connection made succesfully")

except:
    Error
    print(f"connection not made")
