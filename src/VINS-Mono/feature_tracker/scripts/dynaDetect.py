#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from DDRNet.demo import seg

class dynaDetect():
    def __init__(self):
        self.pubMask = rospy.Publisher("/mask", Image, queue_size=10)
        self.subImage = rospy.Subscriber("/image", Image, self.imageCallback)
    
    def imageCallback(self, image_msg):
        print("receive image msg")

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        print("code start")
        rospy.init_node("dynaDetect", anonymous= True)
        dd = dynaDetect()
        dd.spin()
    except rospy.ROSInterruptException:
        print("code error")

'''
#!/usr/bin/env python
     1 #!/usr/bin/env python
   2 import rospy
   3 from std_msgs.msg import String
   4 
   5 def callback(data):
   6     rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
   7     
   8 def listener():
   9 
  10     # In ROS, nodes are uniquely named. If two nodes with the same
  11     # name are launched, the previous one is kicked off. The
  12     # anonymous=True flag means that rospy will choose a unique
  13     # name for our 'listener' node so that multiple listeners can
  14     # run simultaneously.
  15     rospy.init_node('listener', anonymous=True)
  16 
  17     rospy.Subscriber("chatter", String, callback)
  18 
  19     # spin() simply keeps python from exiting until this node is stopped
  20     rospy.spin()
  21 
  22 if __name__ == '__main__':
  23     listener()
'''