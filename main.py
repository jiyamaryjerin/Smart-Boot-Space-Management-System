from py3dbp import Packer, Bin, Item
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st 
import plotly.express as px

from streamlit_option_menu import option_menu
from py3dbp import Packer, Bin, Item
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from py3dbp import Packer, Bin, Item
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import pandas as a
from py3dbp import Packer, Bin, Item
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
 
st.set_page_config(page_title="Boot Space Optimiser", page_icon="💼" )
st.image("bootlogo.png")
 
options=option_menu(
	menu_title=None,
	options=['Home', 'Get Started', 'Visualisation'],
	orientation="horizontal")


def homepage():
	st.title("Welcome to Bootspace Optimiser!")
	st.write("Tired of having to leave bags behind while travelling due to insufficient boot space? Bootspace optimizer helps you utilise your bootspace efficiently. All you need is a picture of your boot and luggage.")

	st.image("bags.png")

	st.write("Ready to try it out.")
	st.write("Click on Get Started")
		 
	


def getstarted():
	
	st.subheader("Upload image of boot space")	
	up_file_boot=st.file_uploader("    ", type=["png","jpeg"])
	photo_boot=st.camera_input("    ")
	if up_file_boot is None and  photo_boot is None:
		print(" ")
	elif photo_boot is None:
		st.image(up_file_boot)
		boot(up_file_boot)
		
		
	else:
		st.image(photo_boot)
		boot(photo_boot)
		
	
	st.subheader("Upload front view image of the luggage")
	up_file_front=st.file_uploader("     ", type=["png","jpeg"])
	photo_front=st.camera_input("       ")
	if up_file_front is None and  photo_front is None:
		print(" ")
	elif photo_front is None:
		st.image(up_file_front)
		img1(up_file_front)
		
	else:
		st.image(photo_front)
		img1(photo_front)
	
	st.subheader("Upload side view image the luggage")
	up_file_side=st.file_uploader("   ", type=["png","jpeg"])
	photo_side=st.camera_input("   ")
	if up_file_side is None and  photo_side is None:
		print("  ")
	elif photo_side is None:
		st.image(up_file_side)
		img2(up_file_side)
		
	else:
		st.image(photo_side)
		img2(photo_side)
import pandas as pd
import cv2
import numpy as np
from object_detector import *
def picture():
	def img1(a):
	    #img 1
	    img=cv2.imread(a)
	    detector=HomogeneousBgDetector()
	    contours= detector.detect_objects(img)
	    for cnt in contours:
	        rect=cv2.minAreaRect(cnt)
	        (x,y),(w,h),angle=rect
	        #print(w/4,"cm")
	        #print(h/4,"cm")
	        box=cv2.boxPoints(rect)
	        box=np.int0(box) 
	        background = np.full((300,600,3), 0, dtype=np.uint8)
	        cv2.circle(img,(int(x),int(y)),5,(0,0,255),-1)
	        cv2.polylines(img, [box],True,(255,0,0),2)
	        x1,y1,w1,h1 = x-50,y-15,175,75
	        cv2.putText(img,"Width{} cm".format(w/4,1),(int(x-50),int(y-15)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)
	        cv2.putText(img,"Height{}cm".format(round(h/4,1)),(int(x-50),int(y+15)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)
	    cv2.imshow("image",img)
	    cv2.waitKey(0)
	    #print("exiting img1")
	    dim['fw'].append(w/4)
	    print(type(h))
	    dim['fh'].append(h/4)

	def img2(b):
	    #pic 2
	    img1=cv2.imread(b)
	    detector=HomogeneousBgDetector()
	    contours= detector.detect_objects(img1)
	    for cnt in contours:
	        rect=cv2.minAreaRect(cnt)
	        (x,y),(d,h),angle=rect
	        #print(d/4,"cm")
	        box=cv2.boxPoints(rect)
	        box=np.int0(box) 
	        background = np.full((300,600,3), 0, dtype=np.uint8)
	        cv2.circle(img1,(int(x),int(y)),5,(0,0,255),-1)
	        cv2.polylines(img1, [box],True,(255,0,0),2)
	        x1,y1,w1,h1 = x-50,y-15,175,75
	        cv2.putText(img1,"Depth{} cm".format(d/4,1),(int(x-50),int(y-15)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)
	        #cv2.putText(img1,"Height{}cm".format(round(h/0.025,1)),(int(x-50),int(y+15)),cv2.FONT_HERSHEY_PLAIN,1,(255, 255, 255),2)
	    cv2.imshow("image",img1)
	    cv2.waitKey(0)
	    #print("img2")
	    dim['sd'].append(d/4)

	def boot(c):
	    ##### boot space
	    #img 1
	    import statistics
	    height_l=[]
	    width_l=[]
	    image=cv2.imread(c)
	    img = cv2.resize(image, (640, 480))
	    detector=HomogeneousBgDetector()
	    contours= detector.detect_objects(img)
	    for cnt in contours:
	        rect=cv2.minAreaRect(cnt)
	        (x,y),(w,h),angle=rect
	        #print("width",w/4,"cm")
	        #print("height",h/4,"cm")
	        box=cv2.boxPoints(rect)
	        box=np.int0(box) 
	        background = np.full((300,600,3), 0, dtype=np.uint8)
	        cv2.circle(img,(int(x),int(y)),5,(0,0,255),-1)
	        cv2.polylines(img, [box],True,(255,0,0),2)
	        x1,y1,w1,h1 = x-50,y-15,175,75
	        cv2.putText(img,"Width{} cm".format(w/4,1),(int(x-50),int(y-15)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)
	        cv2.putText(img,"Height{}cm".format(round(h/4,1)),(int(x-50),int(y+15)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)
	        height_l.append((h/4))
	        width_l.append((w/4))
	    # Calculate the median of the list
	    median = statistics.median(height_l)
	    
	    # Print the median
	    #print("Est Height:", median)
	    
	    #print("LIST ", height_l)
	    #print((height_l[-1]-height_l[-2])*60)
	    #print("LIST ", width_l)
	    #print("Est Width=",max(width_l))
	    cv2.imshow("image",img)
	    cv2.waitKey(0)
	    #print("boot")
	    dim['bw'].append(max(width_l))
	    dim['bh'].append(median)

	a="C:/Users/jiyamary/Desktop/hackathon/streamlit/at1.jpg"
	b="C:/Users/jiyamary/Desktop/hackathon/streamlit/at2.jpg"
	c="C:/Users/jiyamary/Desktop/hackathon/streamlit/i10.jpg"
	#front view- fw/4,fh/4= img1
	#side view- sd/4= img2
	#boot- width=max(width_l), height=median= img3

	dim={'fw':[],'fh':[],'sd':[],'bw':[],'bh':[]}
	img1(a)
	img2(b)
	boot(c)
	print(dim)
	df= pd.DataFrame(dim)
	 
	df.to_csv('C:/Users/jiyamary/Desktop/hackathon/streamlit/samplee.csv')

	

	df= pd.read_csv('C:/Users/jiyamary/Desktop/hackathon/streamlit/samplee.csv')
	length=0
	height=0
	width=0
	for i in df:
	    for j in df['fw']:
	        width= j
	    for j in df['fh']:
	        height=j
	    for j in df['sd']:
	        length=j
	            
	item_data = [
	    {"name": "Swissgear Laptop Backpack", "length": length, "width": width, "height": height, "weight": 0},
	    {"name": "Travelpro Crew 11 Duffel Bag", "length": length, "width": width, "height": height, "weight": 0},
	    {"name": "Briggs & Riley Cabin Bag", "length": length, "width": width, "height": height, "weight": 0}
	    #{"name": "Travelon Underseat Bag", "length": 14, "width": 12, "height": 8, "weight": 0},
	    #{"name": "Travelpro Platinum Elite Duffel Bag", "length": 10, "width": 18, "height": 8, "weight": 0},
	    #{"name": "Reaction Kenneth Cole Underseat Spinner", "length": 17, "width": 13, "height": 9, "weight": 0},
	]
	print(item_data)
	    
	storage_unit = Bin('StorageUnit',50,123,25,0)

	packer = Packer()

	# Add the storage unit to the packer
	packer.add_bin(storage_unit)

	# Add items to the packer dynamically
	for item in item_data:
	    new_item = Item(item["name"], item["length"], item["width"], item["height"], item["weight"])
	    packer.add_item(new_item)
	    print(f"Added item: {item['name']} with dimensions {item['length']}x{item['width']}x{item['height']}")

	# Run the packing algorithm
	packer.pack()

	for bin in packer.bins:
	    print("Items in bin:", bin.name)
	    for item in bin.items:
	        print(f" Item: {item.name} at position {item.position} with dimensions {item.get_dimension()}")

	def get_random_color():
	    return np.random.rand(3,)

	color_mapping = {}

	# Function to add a 3D box (representing an item)
	def add_box(ax, item, color):
	    # Extracting position and dimensions
	    pos = np.array(item.position, dtype=float)
	    dim = np.array(item.get_dimension(), dtype=float)

	    # Create a rectangular prism
	    xx, yy = np.meshgrid([pos[0], pos[0]+dim[0]], [pos[1], pos[1]+dim[1]])
	    ax.plot_surface(xx, yy, np.full_like(xx, pos[2]), color=color, alpha=0.5)
	    ax.plot_surface(xx, yy, np.full_like(xx, pos[2]+dim[2]), color=color, alpha=0.5)
	   
	    yy, zz = np.meshgrid([pos[1], pos[1]+dim[1]], [pos[2], pos[2]+dim[2]])
	    ax.plot_surface(np.full_like(yy, pos[0]), yy, zz, color=color, alpha=0.5)
	    ax.plot_surface(np.full_like(yy, pos[0]+dim[0]), yy, zz, color=color, alpha=0.5)
	   
	    xx, zz = np.meshgrid([pos[0], pos[0]+dim[0]], [pos[2], pos[2]+dim[2]])
	    ax.plot_surface(xx, np.full_like(xx, pos[1]), zz, color=color, alpha=0.5)
	    ax.plot_surface(xx, np.full_like(xx, pos[1]+dim[1]), zz, color=color, alpha=0.5)

	# Create a 3D plot
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# Adding each item in the storage unit to the plot dynamically
	#for item in storage_unit.items:
	#    color = get_random_color()  # Assign a random color to each item
	#    add_box(ax, item, color)

	# Adding each item in the storage unit to the plot
	for item in storage_unit.items:
	    color = get_random_color()  # Get a random color for each item
	    add_box(ax, item, color)
	    color_mapping[item.name] = color  # Store the color mapping

	# Create a legend/key for the items
	legend_labels = [plt.Line2D([0], [0], color=color, lw=4, label=name) for name, color in color_mapping.items()]
	plt.legend(handles=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

	# Setting the limits to match the storage unit size
	ax.set_xlim([0, 50])
	ax.set_ylim([0, 123])
	ax.set_zlim([0, 25])

	# Labels and title
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')
	ax.set_title('3D Visualization of Furniture in Storage Unit')

	plt.show()


	storage_unit = Bin('StorageUnit',13,52,36,0)

	item_data = [
	    {"name": "Swissgear Laptop Backpack", "length": 14, "width": 10, "height": 8, "weight": 0},
	    {"name": "Travelpro Crew 11 Duffel Bag", "length": 11, "width": 16, "height": 8, "weight": 0},
	    {"name": "Briggs & Riley Cabin Bag", "length": 17, "width": 9, "height": 7, "weight": 0},
	    {"name": "Travelon Underseat Bag", "length": 14, "width": 12, "height": 8, "weight": 0},
	    {"name": "Travelpro Platinum Elite Duffel Bag", "length": 10, "width": 18, "height": 8, "weight": 0},
	    {"name": "Reaction Kenneth Cole Underseat Spinner", "length": 17, "width": 13, "height": 9, "weight": 0},
	]

	packer = Packer()

	# Add the storage unit to the packer
	packer.add_bin(storage_unit)

	# Add items to the packer dynamically
	for item in item_data:
	    new_item = Item(item["name"], item["length"], item["width"], item["height"], item["weight"])
	    packer.add_item(new_item)
	    print(f"Added item: {item['name']} with dimensions {item['length']}x{item['width']}x{item['height']}")

	# Run the packing algorithm
	packer.pack()

	for bin in packer.bins:
	    print("Items in bin:", bin.name)
	    for item in bin.items:
	        print(f" Item: {item.name} at position {item.position} with dimensions {item.get_dimension()}")

	def get_random_color():
	    return np.random.rand(3,)

	color_mapping = {}

	# Function to add a 3D box (representing an item)
	def add_box(ax, item, color):
	    # Extracting position and dimensions
	    pos = np.array(item.position, dtype=float)
	    dim = np.array(item.get_dimension(), dtype=float)

	    # Create a rectangular prism
	    xx, yy = np.meshgrid([pos[0], pos[0]+dim[0]], [pos[1], pos[1]+dim[1]])
	    ax.plot_surface(xx, yy, np.full_like(xx, pos[2]), color=color, alpha=0.5)
	    ax.plot_surface(xx, yy, np.full_like(xx, pos[2]+dim[2]), color=color, alpha=0.5)
	   
	    yy, zz = np.meshgrid([pos[1], pos[1]+dim[1]], [pos[2], pos[2]+dim[2]])
	    ax.plot_surface(np.full_like(yy, pos[0]), yy, zz, color=color, alpha=0.5)
	    ax.plot_surface(np.full_like(yy, pos[0]+dim[0]), yy, zz, color=color, alpha=0.5)
	   
	    xx, zz = np.meshgrid([pos[0], pos[0]+dim[0]], [pos[2], pos[2]+dim[2]])
	    ax.plot_surface(xx, np.full_like(xx, pos[1]), zz, color=color, alpha=0.5)
	    ax.plot_surface(xx, np.full_like(xx, pos[1]+dim[1]), zz, color=color, alpha=0.5)

	# Create a 3D plot
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# Adding each item in the storage unit to the plot dynamically
	#for item in storage_unit.items:
	#    color = get_random_color()  # Assign a random color to each item
	#    add_box(ax, item, color)

	# Adding each item in the storage unit to the plot
	for item in storage_unit.items:
	    color = get_random_color()  # Get a random color for each item
	    add_box(ax, item, color)
	    color_mapping[item.name] = color  # Store the color mapping

	# Create a legend/key for the items
	legend_labels = [plt.Line2D([0], [0], color=color, lw=4, label=name) for name, color in color_mapping.items()]
	plt.legend(handles=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

	# Setting the limits to match the storage unit size
	ax.set_xlim([0, 13])
	ax.set_ylim([0, 52])
	ax.set_zlim([0, 36])

	# Labels and title
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')
	ax.set_title('3D Visualization of Furniture in Storage Unit')

	plt.show()




	storage_unit = Bin('StorageUnit',13,52,36,0)

	item_data = [
	    {"name": "Swissgear Laptop Backpack", "length": 14, "width": 10, "height": 8, "weight": 0},
	    {"name": "Travelpro Crew 11 Duffel Bag", "length": 11, "width": 16, "height": 8, "weight": 0},
	    {"name": "Briggs & Riley Cabin Bag", "length": 17, "width": 9, "height": 7, "weight": 0},
	    {"name": "Travelon Underseat Bag", "length": 14, "width": 12, "height": 8, "weight": 0},
	    {"name": "Travelpro Platinum Elite Duffel Bag", "length": 10, "width": 18, "height": 8, "weight": 0},
	    {"name": "Reaction Kenneth Cole Underseat Spinner", "length": 17, "width": 13, "height": 9, "weight": 0},
	]

	packer = Packer()

	# Add the storage unit to the packer
	packer.add_bin(storage_unit)

	# Add items to the packer dynamically
	for item in item_data:
	    new_item = Item(item["name"], item["length"], item["width"], item["height"], item["weight"])
	    packer.add_item(new_item)
	    print(f"Added item: {item['name']} with dimensions {item['length']}x{item['width']}x{item['height']}")

	# Run the packing algorithm
	packer.pack()

	for bin in packer.bins:
	    print("Items in bin:", bin.name)
	    for item in bin.items:
	        print(f" Item: {item.name} at position {item.position} with dimensions {item.get_dimension()}")

	def get_random_color():
	    return np.random.rand(3,)

	color_mapping = {}

	# Function to add a 3D box (representing an item)
	def add_box(ax, item, color):
	    # Extracting position and dimensions
	    pos = np.array(item.position, dtype=float)
	    dim = np.array(item.get_dimension(), dtype=float)

	    # Create a rectangular prism
	    xx, yy = np.meshgrid([pos[0], pos[0]+dim[0]], [pos[1], pos[1]+dim[1]])
	    ax.plot_surface(xx, yy, np.full_like(xx, pos[2]), color=color, alpha=0.5)
	    ax.plot_surface(xx, yy, np.full_like(xx, pos[2]+dim[2]), color=color, alpha=0.5)
	   
	    yy, zz = np.meshgrid([pos[1], pos[1]+dim[1]], [pos[2], pos[2]+dim[2]])
	    ax.plot_surface(np.full_like(yy, pos[0]), yy, zz, color=color, alpha=0.5)
	    ax.plot_surface(np.full_like(yy, pos[0]+dim[0]), yy, zz, color=color, alpha=0.5)
	   
	    xx, zz = np.meshgrid([pos[0], pos[0]+dim[0]], [pos[2], pos[2]+dim[2]])
	    ax.plot_surface(xx, np.full_like(xx, pos[1]), zz, color=color, alpha=0.5)
	    ax.plot_surface(xx, np.full_like(xx, pos[1]+dim[1]), zz, color=color, alpha=0.5)

	# Create a 3D plot
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# Adding each item in the storage unit to the plot dynamically
	#for item in storage_unit.items:
	#    color = get_random_color()  # Assign a random color to each item
	#    add_box(ax, item, color)

	# Adding each item in the storage unit to the plot
	for item in storage_unit.items:
	    color = get_random_color()  # Get a random color for each item
	    add_box(ax, item, color)
	    color_mapping[item.name] = color  # Store the color mapping

	# Create a legend/key for the items
	legend_labels = [plt.Line2D([0], [0], color=color, lw=4, label=name) for name, color in color_mapping.items()]
	plt.legend(handles=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

	# Setting the limits to match the storage unit size
	ax.set_xlim([0, 13])
	ax.set_ylim([0, 52])
	ax.set_zlim([0, 36])

	# Labels and title
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')
	ax.set_title('3D Visualization of Furniture in Storage Unit')

	plt.show()




	storage_unit = Bin('StorageUnit',13,52,36,0)

	item_data = [
	    {"name": "Swissgear Laptop Backpack", "length": 14, "width": 10, "height": 8, "weight": 0},
	    {"name": "Travelpro Crew 11 Duffel Bag", "length": 11, "width": 16, "height": 8, "weight": 0},
	    {"name": "Briggs & Riley Cabin Bag", "length": 17, "width": 9, "height": 7, "weight": 0},
	    {"name": "Travelon Underseat Bag", "length": 14, "width": 12, "height": 8, "weight": 0},
	    {"name": "Travelpro Platinum Elite Duffel Bag", "length": 10, "width": 18, "height": 8, "weight": 0},
	    {"name": "Reaction Kenneth Cole Underseat Spinner", "length": 17, "width": 13, "height": 9, "weight": 0},
	]

	packer = Packer()

	# Add the storage unit to the packer
	packer.add_bin(storage_unit)

	# Add items to the packer dynamically
	for item in item_data:
	    new_item = Item(item["name"], item["length"], item["width"], item["height"], item["weight"])
	    packer.add_item(new_item)
	    print(f"Added item: {item['name']} with dimensions {item['length']}x{item['width']}x{item['height']}")

	# Run the packing algorithm
	packer.pack()

	for bin in packer.bins:
	    print("Items in bin:", bin.name)
	    for item in bin.items:
	        print(f" Item: {item.name} at position {item.position} with dimensions {item.get_dimension()}")

	def get_random_color():
	    return np.random.rand(3,)

	color_mapping = {}

	# Function to add a 3D box (representing an item)
	def add_box(ax, item, color):
	    # Extracting position and dimensions
	    pos = np.array(item.position, dtype=float)
	    dim = np.array(item.get_dimension(), dtype=float)

	    # Create a rectangular prism
	    xx, yy = np.meshgrid([pos[0], pos[0]+dim[0]], [pos[1], pos[1]+dim[1]])
	    ax.plot_surface(xx, yy, np.full_like(xx, pos[2]), color=color, alpha=0.5)
	    ax.plot_surface(xx, yy, np.full_like(xx, pos[2]+dim[2]), color=color, alpha=0.5)
	   
	    yy, zz = np.meshgrid([pos[1], pos[1]+dim[1]], [pos[2], pos[2]+dim[2]])
	    ax.plot_surface(np.full_like(yy, pos[0]), yy, zz, color=color, alpha=0.5)
	    ax.plot_surface(np.full_like(yy, pos[0]+dim[0]), yy, zz, color=color, alpha=0.5)
	   
	    xx, zz = np.meshgrid([pos[0], pos[0]+dim[0]], [pos[2], pos[2]+dim[2]])
	    ax.plot_surface(xx, np.full_like(xx, pos[1]), zz, color=color, alpha=0.5)
	    ax.plot_surface(xx, np.full_like(xx, pos[1]+dim[1]), zz, color=color, alpha=0.5)

	# Create a 3D plot
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# Adding each item in the storage unit to the plot dynamically
	#for item in storage_unit.items:
	#    color = get_random_color()  # Assign a random color to each item
	#    add_box(ax, item, color)

	# Adding each item in the storage unit to the plot
	for item in storage_unit.items:
	    color = get_random_color()  # Get a random color for each item
	    add_box(ax, item, color)
	    color_mapping[item.name] = color  # Store the color mapping

	# Create a legend/key for the items
	legend_labels = [plt.Line2D([0], [0], color=color, lw=4, label=name) for name, color in color_mapping.items()]
	plt.legend(handles=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

	# Setting the limits to match the storage unit size
	ax.set_xlim([0, 13])
	ax.set_ylim([0, 52])
	ax.set_zlim([0, 36])

	# Labels and title
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')
	ax.set_title('3D Visualization of Furniture in Storage Unit')

	plt.show()

	 

def plotting():
	storage_unit = Bin('StorageUnit',13,52,36,0)

	item_data = [
    	{"name": "Swissgear Laptop Backpack", "length": 14, "width": 10, "height": 8, "weight": 0},
    	{"name": "Travelpro Crew 11 Duffel Bag", "length": 11, "width": 16, "height": 8, "weight": 0},
    	{"name": "Briggs & Riley Cabin Bag", "length": 17, "width": 9, "height": 7, "weight": 0},
    	{"name": "Travelon Underseat Bag", "length": 14, "width": 12, "height": 8, "weight": 0},
    	{"name": "Travelpro Platinum Elite Duffel Bag", "length": 10, "width": 18, "height": 8, "weight": 0},
    	{"name": "Reaction Kenneth Cole Underseat Spinner", "length": 17, "width": 13, "height": 9, "weight": 0},
	]

	packer = Packer()

	# Add the storage unit to the packer
	packer.add_bin(storage_unit)

	# Add items to the packer dynamically
	for item in item_data:
		new_item = Item(item["name"], item["length"], item["width"], item["height"], item["weight"])
		packer.add_item(new_item)
    	
	# Run the packing algorithm
	packer.pack()

	

	def get_random_color():
		return np.random.rand(3,)

	color_mapping = {}

	# Function to add a 3D box (representing an item)
	def add_box(ax, item, color):
    	# Extracting position and dimensions
		pos = np.array(item.position, dtype=float)
		dim = np.array(item.get_dimension(), dtype=float)

 	   # Create a rectangular prism
		xx, yy = np.meshgrid([pos[0], pos[0]+dim[0]], [pos[1], pos[1]+dim[1]])
		ax.plot_surface(xx, yy, np.full_like(xx, pos[2]), color=color, alpha=0.5)
		ax.plot_surface(xx, yy, np.full_like(xx, pos[2]+dim[2]), color=color, alpha=0.5)
   
		yy, zz = np.meshgrid([pos[1], pos[1]+dim[1]], [pos[2], pos[2]+dim[2]])
		ax.plot_surface(np.full_like(yy, pos[0]), yy, zz, color=color, alpha=0.5)
		ax.plot_surface(np.full_like(yy, pos[0]+dim[0]), yy, zz, color=color, alpha=0.5)
   
		xx, zz = np.meshgrid([pos[0], pos[0]+dim[0]], [pos[2], pos[2]+dim[2]])
		ax.plot_surface(xx, np.full_like(xx, pos[1]), zz, color=color, alpha=0.5)
		ax.plot_surface(xx, np.full_like(xx, pos[1]+dim[1]), zz, color=color, alpha=0.5)

	# Create a 3D plot
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

		# Adding each item in the storage unit to the plot dynamically
		#for item in storage_unit.items:
		#    color = get_random_color()  # Assign a random color to each item
		#    add_box(ax, item, color)

	# Adding each item in the storage unit to the plot
	for item in storage_unit.items:
		color = get_random_color()  # Get a random color for each item
		add_box(ax, item, color)
		color_mapping[item.name] = color  # Store the color mapping

	# Create a legend/key for the items
	legend_labels = [plt.Line2D([0], [0], color=color, lw=4, label=name) for name, color in color_mapping.items()]
	plt.legend(handles=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

	# Setting the limits to match the storage unit size
	ax.set_xlim([0, 13])
	ax.set_ylim([0, 52])
	ax.set_zlim([0, 36])

	# Labels and title
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')
	ax.set_title('3D Visualization of Luggage in Bootspace')

	plt.show()
	plt.savefig('graph.png', dpi=300)
	st.image("graph.png")	
	
if options=='Home':
	homepage()

elif options=='Get Started':
	getstarted()
elif options=='Visualisation':
	picture()
	plotting()