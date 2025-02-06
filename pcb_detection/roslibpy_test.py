import roslibpy

# https://roslibpy.readthedocs.io/en/latest/examples.html
# https://docs.pal-robotics.com/tiago-single/handbook.html

client = roslibpy.Ros(host='tiago-158c', port=9090)

###

client.run()
print('Is ROS connected?', client.is_connected)

###

#client.on_ready(lambda: print('Is ROS connected?', client.is_connected))
#client.run_forever()

###

client.terminate()