import random
import hashlib #cryptographic hashes (consistency)
import noise #for more realistic generation
import math
from noise import pnoise2
#Array Visualization
import matplotlib.pyplot as plt
import numpy as np
#VORONOI SEEDING
from matplotlib.colors import ListedColormap, BoundaryNorm

from scipy.spatial import Voronoi,voronoi_plot_2d

#LIST OF WORDS FOR RANDOMIZED SEEDS
from nltk.corpus import words as nltk_words
WORLD_LIST = nltk_words
#FAULT LINE
from skimage.draw import line
#FAULT LINE FALL OFF FOR TERRAIN
from scipy.ndimage import distance_transform_edt

#RESOURCE MGMT
from collections import defaultdict

from faker import Faker




# DISPLAY
def write_world_to_file(seed_as_string,world_map,file_name,print_type):
    
    
    #print_type = "value" or "symbol"

    min_val = min(min(row) for row in world_map)
    max_val = max(max(row) for row in world_map)
    
    height_range = (max_val - min_val)
    water_threshold = height_range * 0.1 + min_val
    grass_threshold = height_range * 0.5 + min_val
    forest_threshold = height_range * 0.9 + min_val
    mountain_threshold = height_range + min_val
    
    terrain_symbols = {
    "water": "~",
    "grass": "#",
    "forest":"#",
    "mountain": "^"
    }
    file = open(file_name + ".txt","w")
    
    WORLD_SEED = seed_from_string(seed_as_string)

    file.write(f"Seed: {seed_as_string} ({WORLD_SEED})\n")
    file.write (f"MIN: {min_val} MAX: {max_val}\nWATER: {water_threshold} GRASS: {grass_threshold} FOREST: {forest_threshold} MTN: {mountain_threshold}\n")
    size = len(world_map[0])
    for row in range(size):
        for col in range(size):
            
            if (print_type == "value"):
                file.write(f"{world_map[row][col]:.2f} ")
                continue
            if (print_type == "symbol"):
                if (world_map[row][col] < water_threshold):
                    file.write (terrain_symbols["water"])
                    continue
                elif (world_map[row][col] < grass_threshold):
                    file.write (terrain_symbols["grass"])
                    continue
                elif (world_map[row][col] < forest_threshold):
                    file.write (terrain_symbols["forest"])
                    continue
                elif (world_map[row][col] < mountain_threshold):
                    file.write (terrain_symbols["mountain"])
                    continue
                else:
                    file.write ("z")

            
        file.write("\n")
    file.close()
    return
def display_world_altitude(world_map):
    terrain_symbols = {
    "water": "~",
    "mountain": "^",
    "grass": "-",
    "forest":"#"
    }
    height_symbol_dict = {}
    for i in range(10):
        key = float(i)/10
        height_symbol_dict[key] = i
    
    size = len(world_map[0])
    for row in range(size):
        for col in range(size):
            print(f"{world_map[row][col]:.9f} ", end="")
            
        print("\n")
    return
def display_world_GUI(world_map,SEED_AS_STRING):
    size = len(world_map[0])
    vmin = np.min(world_map)
    vmax = np.max(world_map)
    vrange = vmax - vmin
    print(vmin, " ",vmax)
    water_threshold = (vmin + (vrange*0.4))
    sand_threshold = (vmin + (vrange*0.44))
    grass_threshold = (vmin + (vrange*0.6))
    lower_Mtn_threshold = (vmin + (vrange*0.9))

    bounds = [vmin, water_threshold, sand_threshold, grass_threshold,lower_Mtn_threshold, vmax]  # strictly increasing!

    # Define matching colors
    colors = [
        "#0000cc",   # Blue for < 0
        "#C2B280",   # tan
        "#228B22",   # Green
        "#707070",   # Gray
        "#ffffff"    # White
    ]

    # Create colormap and norm
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, len(colors))

    # Clamp world_map to valid range for bounds (just in case)
    clamped_map = np.clip(world_map, -1.0, 1.0)

    # Plot - OLD
    # img = plt.imshow(clamped_map, cmap=cmap, norm=norm, interpolation="nearest")
    # cbar = plt.colorbar(img, ticks=bounds)
    # cbar.set_label("Elevation")

    
    fig, ax = plt.subplots() #fig = orig map, ax = cropped, viewing portion
    img = ax.imshow(clamped_map, cmap=cmap, norm=norm, interpolation="nearest")
    cbar = plt.colorbar(img, ax=ax, ticks=bounds)
    cbar.set_label("Elevation")
    # Initial camera position and zoom
    zoom_radius = 30
    center_x, center_y = size // 2, size // 2

    def update_view():
        ax.set_xlim(center_x - zoom_radius, center_x + zoom_radius)
        ax.set_ylim(center_y + zoom_radius, center_y - zoom_radius)  # y-axis inverted
        fig.canvas.draw_idle()
    def on_key(event):
        nonlocal center_x, center_y, zoom_radius
        step = 5
        zoom_step = 5

        if event.key == 'left':
            center_x -= step
        elif event.key == 'right':
            center_x += step
        elif event.key == 'up':
            center_y -= step
        elif event.key == 'down':
            center_y += step
        elif event.key == '+' or event.key == '=': #done for mac users
            zoom_radius = max(5, zoom_radius - zoom_step)
        elif event.key == '-':
            zoom_radius = min(size // 2, zoom_radius + zoom_step)

        # Clamp to world bounds
        center_x = max(zoom_radius, min(center_x, size - zoom_radius))
        center_y = max(zoom_radius, min(center_y, size - zoom_radius))

        update_view()

    fig.canvas.mpl_connect('key_press_event', on_key)
    ax.set_title(f"World: {SEED_AS_STRING}  \nArrows to pan, +/- to zoom")
    update_view()

    try:
        fig.canvas.manager.window.activateWindow()
        fig.canvas.manager.window.raise_()
    except:
        pass
    #plt.show()
    # plt.title(SEED_AS_STRING)
    # plt.show()
    # plt.imshow(world_map, cmap="gist_earth", interpolation='nearest',alpha =1.0,vmin = -1,vmax = 1)
    # plt.colorbar()
    # plt.show()
    return

# SEEDING
def seed_from_string(s):
    # Convert the string into a 32-bit integer using SHA-256
    # whatever string inputted will always return the same randomized seed
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**32)
    # USAGE:
    #   seed = seed_from_string("BananasApplesHotDogWatermelon")
    #   random.seed(seed)
    #   print(random.randint(0, 100)) WILL ALWAYS RETURN THE SAME NUMBER PER STRING
def create_seed():
    seed = ""
    trueRandom = random.SystemRandom()
    random_number = trueRandom.randint(1,10)
    for _ in range(random_number):
        seed += trueRandom.choice(nltk_words.words()).capitalize()
    return seed

# TECTONIC PLATES
#Vor_regions = 2D Arr holding plate identity
def Voronoi_seeding(size,seed_density,WORLD_SEED):
    # rng = random.Random(WORLD_SEED)
    num_seeds = max(int((size**2) * seed_density),4) #Vor requires 4 seeds minimum
    vor_points = [[rng.uniform(0, size), rng.uniform(0, size)] for _ in range(num_seeds)]

    vor_points = np.array(vor_points,dtype=float) #converting to NumPy array (for [:,0] & [:,1])
    # plt.scatter(vor_points[:,0],vor_points[:,1])
    if len(vor_points) == 0: print("No seed points generated")
    if len(vor_points > 3):
        vor_object = Voronoi(vor_points)
        vor_vertices = vor_object.vertices
        vor_regions = vor_object.regions
        # voronoi_plot_2d(vor_object,show_vertices=False,line_colors = 'blue')
    # plt.title("Voronoi Seeding")

    vor_regions = np.zeros((size, size), dtype=int)

    for col in range(size):
        for row in range(size):
            min_dist = float('inf')
            min_ID = 0
            for ID,seed in enumerate(vor_points):
                sx, sy = seed[0],seed[1]
                dist = ((sy - row)**2) + ((sx-col)**2)
                if dist < min_dist:
                    min_dist = dist
                    min_ID = ID
            vor_regions[row][col] = min_ID

    # plt.figure("Region")
    # plt.imshow(vor_regions,cmap="gray",label="Regions")
    # plt.colorbar()
    # plt.title("Generated Voronoi Regions")
    
    return vor_points,vor_regions
def create_fault_lines(size, vor_regions,plate_list,vor_seeds): #returns binary value map of borders 
    is_vor_border = np.zeros((size, size), dtype=int)
    def create_adjacency(size, vor_regions):
        adjacency_dict = {}
        for row in range(size):
            for col in range(size):
                center_plate= vor_regions[row][col]
                if center_plate not in adjacency_dict:
                    adjacency_dict[center_plate] = set()
                for dy,dx in [(+1,0),(-1,0),(0,+1),(0,-1)]:
                    ny, nx = dy + row, dx + col
                    if (0<=ny<size) and (0<=nx<size):
                        neighbor_plate = vor_regions[ny][nx]
                        if neighbor_plate != center_plate:
                            adjacency_dict[center_plate].add(neighbor_plate)
        adjacency = {k: sorted(list(v)) for k, v in adjacency_dict.items()}
        return adjacency
    adj_table = create_adjacency(size,vor_regions)

    for row in range(size):
        for col in range(size):
            if (row > 0 and row < (size - 1)): # ensures within vert bounds
                if (vor_regions[row][col] != vor_regions[row+1][col]) or (vor_regions[row][col] != vor_regions[row-1][col]): is_vor_border[row][col] = 1
            if (col > 0 and col < (size - 1)):
                if (vor_regions[row][col] != vor_regions[row][col-1]) or (vor_regions[row][col] != vor_regions[row][col+1]): is_vor_border[row][col] = 1

    #Identify fault type
    fault_id_table = {}
    for plate_id in adj_table:
        drift1 = plate_list[plate_id]["drift"]
        for neighbors in adj_table[plate_id]:
            if plate_id not in fault_id_table:
                fault_id_table[plate_id] = {}
            if neighbors in fault_id_table[plate_id]:
                continue
            drift2 = plate_list[neighbors]["drift"]
            relative_drift = np.array(drift2) - np.array(drift1) 
            c1x,c1y = vor_seeds[plate_id]
            c2x,c2y = vor_seeds[neighbors]
            direction_vector = np.array([c2x - c1x, c2y - c1y])
            norm = np.linalg.norm(direction_vector)
            if norm == 0: continue
            direction_vector = direction_vector / norm
            projected_speed = np.dot(relative_drift,direction_vector) #difference of plates at the angle of the plate collision

            
            if projected_speed > 0.4: #2
                fault_id_table[plate_id][neighbors] = 2 # CONVERGENT
            elif projected_speed < -0.4: #-2
                fault_id_table[plate_id][neighbors] = -2 # DIVERGENT
            else:
                fault_id_table[plate_id][neighbors] = 1 # OTHER... Setting it for recongition/testing

    fault_map = np.zeros((size,size))
    for row in range(size):
        for col in range(size):
            if is_vor_border[row][col] == 1:
                plate1 = vor_regions[row][col]
                for dy,dx in [[+1,0],[-1,0],[0,+1],[0,-1]]:
                    ny,nx = dy + row,dx + col
                    if (0<=ny<size and 0<=nx<size):
                        plate2 = vor_regions[ny][nx]
                        if plate1 != plate2: #FETCH THE IDENTIFICATION
                            fault_map[row][col] = fault_id_table[plate1][plate2]


    # plt.figure("Boundaries")
    # plt.imshow(is_vor_border,cmap="gray",label="Boundaries")
    # plt.colorbar()
    # plt.title("Generated Voronoi Edges")

    plt.figure("FaultMap ")
    plt.imshow(fault_map,cmap="gray",label="Boundaries")
    plt.colorbar()
    plt.title("Generated Fault Map")
    # plt.show()
    return is_vor_border,fault_map
def create_tectonic_plates(vor_ID,vor_regions,size,WORLD_SEED): # returns dict of plates
    #setting random to WORLD_SEED
    # rng = np.random.default_rng(WORLD_SEED)

    def create_random_unit_vector():
        theta= math.radians( rng.integers(0,361) )
        x = math.cos( theta)
        y = math.sin(theta)
        return (x,y)
    
    tectonic_plate_dict = {}
    for regions_ID in range(len(vor_ID)):
        if (rng.integers(0,101) > 40):
            plate_type = "continental"
            base_elevation = rng.uniform(0.2,0.4)
        else: 
            plate_type = "oceanic"
            base_elevation = rng.uniform(-0.4,-0.2)


        tect_plate_info = {}
        tect_plate_info["drift"] = create_random_unit_vector()
        tect_plate_info["type"] = plate_type
        tect_plate_info["base_elevation"] = base_elevation
        
        tectonic_plate_dict[regions_ID] = tect_plate_info
    #print(tectonic_plate_dict)
    return tectonic_plate_dict
def create_fault_map(vor_ID_list, vor_regions_map, tect_plates, size, WORLD_SEED): 
    vor_obj = Voronoi(vor_ID_list)
    fault_lines_map = np.zeros((size, size), dtype=int)

    def clip(val, minval, maxval):
        return max(minval, min(val, maxval))

    for (p1, p2), vertex_indices in zip(vor_obj.ridge_points, vor_obj.ridge_vertices):
        if len(vertex_indices) != 2:
            continue

        # Finite ridge
        if -1 not in vertex_indices:
            v1 = vor_obj.vertices[vertex_indices[0]]
            v2 = vor_obj.vertices[vertex_indices[1]]
        else:
            # Infinite ridge: project it out
            finite_idx = vertex_indices[1] if vertex_indices[0] == -1 else vertex_indices[0]
            v1 = vor_obj.vertices[finite_idx]

            # Get perpendicular direction from seed points
            point1 = vor_obj.points[p1]
            point2 = vor_obj.points[p2]
            direction = point2 - point1
            direction = np.array([-direction[1], direction[0]])  # perpendicular

            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            direction /= norm

            v2 = v1 + direction * (size * 2)  # Project far enough to cross map

        # Skip if invalid or collapsed
        if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
            continue
        if np.linalg.norm(v2 - v1) < 1e-6:
            continue

        # Convert to clipped pixel coordinates
        r0 = int(clip(round(v1[1]), 0, size - 1))
        c0 = int(clip(round(v1[0]), 0, size - 1))
        r1 = int(clip(round(v2[1]), 0, size - 1))
        c1 = int(clip(round(v2[0]), 0, size - 1))

        rr, cc = line(r0, c0, r1, c1)
        fault_lines_map[rr, cc] = 1

    # Display
    plt.figure("Fault Map")
    plt.imshow(fault_lines_map, cmap="gray")
    plt.title("All Fault Lines (with Infinite Edges)")
    plt.colorbar()
    # plt.show()

    return fault_lines_map
def populate_resources(altitude_map,fault_lines_map,size,temp_type, water_threshold,mtn_threshold,WORLD_SEED):
    #2 = converge
    #-2 = diverge
    #1 = other faultlines

    
    resource_map = np.full((size, size), fill_value=0, dtype=np.int8)
    resources_dict = {
        0: "None",
        1: "Wood",
        2: "Salt",
        3: "Coal",
        4: "Iron",
        5: "Gold", 
        6: "Grain",
        7: "Oil",
        8: "Stone",
    } 
    resources_rarity = {
        "None" : 1, #this determines density of resources in map
        "Stone" : 0.0,
        "Wood" : 0.8,
        "Salt" : 0, #it will be manually raised then lowered for ocean tiles
        "Coal" : 0.1,
        "Iron" : 0.1,
        "Gold" : 0.1,
        "Grain" : 0.9,
        "Oil" : 0.1
    }
    con_resources_rarity = {
        "None" : .6, #this determines density of resources in map
        "Stone" : 2,
        "Wood" : 0.05,
        "Salt" : 0, #it will be manually raised then lowered for ocean tiles
        "Coal" : 0.5,
        "Iron" : 0.6,
        "Gold" : 0.4,
        "Grain" : 0,
        "Oil" : 0.2
    }
    div_resources_rarity = {
        "None" : 1, #this determines density of resources in map
        "Stone" : 0.8,
        "Wood" : 0.0,
        "Salt" : 0.5, #it will be manually raised then lowered for ocean tiles
        "Coal" : 0.5,
        "Iron" : 0.6,
        "Gold" : 0.2,
        "Grain" : 0,
        "Oil" : 0.8
    }
    tran_resources_rarity = {
        "None" : 1, #this determines density of resources in map
        "Stone" : 1,
        "Wood" : 2,
        "Salt" : 0, #it will be manually raised then lowered for ocean tiles
        "Coal" : 0.7,
        "Iron" : 0.6,
        "Gold" : 0.3,
        "Grain" : 3,
        "Oil" : 0.6
    }
    wtr_resources_rarity = {
    
        "None" : 2, #this determines density of resources in map
        "Stone" : 0.0,
        "Wood" : 0.0,
        "Salt" : 0.9, #it will be manually raised then lowered for ocean tiles
        "Coal" : 0,
        "Iron" : 0,
        "Gold" : 0,
        "Grain" : 0.0,
        "Oil" : 0.6

    }
    resource_types = list(resources_rarity.keys())
    
    if temp_type == "hot":
        con_resources_rarity["Salt"] += 0.2
        div_resources_rarity["Salt"] += 0.2
        tran_resources_rarity["Salt"] += 0.2

        con_resources_rarity["Oil"] += 0.2
        div_resources_rarity["Oil"] += 0.2
        tran_resources_rarity["Oil"] += 0.2

        con_resources_rarity["Gold"] += 0.1

    random.seed(WORLD_SEED)
    # rng = random.random()
    threshold = 10
    convergent_mask = (fault_lines_map == 2  ).astype(float)
    dist_div = distance_transform_edt(1 - convergent_mask) # map of float that indicates distance from fault
    divergent_mask = (fault_lines_map == -2  ).astype(float)
    dist_conv = distance_transform_edt(1 - divergent_mask) # map of float that indicates distance from fault
    transform_mask = (fault_lines_map == 1  ).astype(float)
    dist_tran = distance_transform_edt(1 - transform_mask) # map of float that indicates distance from fault
   
    
    weights = list(resources_rarity.values())   
    con_weights = list(con_resources_rarity.values())   
    div_weights = list(div_resources_rarity.values())  
    tran_weights = list(tran_resources_rarity.values())  
    wtr_weights = list(wtr_resources_rarity.values())  

    conv_indices = np.argwhere(dist_conv < threshold)

    used_mask = np.zeros((size, size), dtype=bool)
    for y,x in conv_indices:
        if altitude_map[x][y] <= water_threshold:
            resource_at_tile = random.choices(resource_types, weights=wtr_weights, k=1)[0]
        else:
            resource_at_tile = random.choices(resource_types, weights=con_weights, k=1)[0]  

        resource_ID = key = next((k for k, v in resources_dict.items() if v == resource_at_tile), 0)
        resource_map[x][y] = resource_ID
        used_mask[x][y] = True
    div_indices = np.argwhere(dist_div < threshold)
    for y,x in div_indices:
        if altitude_map[x][y] < water_threshold:
            resource_at_tile = random.choices(resource_types, weights=wtr_weights, k=1)[0]
        else:
            resource_at_tile = random.choices(resource_types, weights=div_weights, k=1)[0] 
        resource_ID = key = next((k for k, v in resources_dict.items() if v == resource_at_tile), 0)
        resource_map[x][y] = resource_ID
        used_mask[x][y] = True
    tran_indices = np.argwhere(dist_tran < threshold)
    for y,x in tran_indices:
        if altitude_map[x][y] < water_threshold:
            resource_at_tile = random.choices(resource_types, weights=wtr_weights, k=1)[0]
        else:
            resource_at_tile = random.choices(resource_types, weights=con_weights, k=1)[0] 
        resource_ID = key = next((k for k, v in resources_dict.items() if v == resource_at_tile), 0)
        resource_map[x][y] = resource_ID
        used_mask[x][y] = True
    
    other_indices = np.argwhere(used_mask == False)
    for y,x in other_indices:
        if altitude_map[x][y] < water_threshold:
            resource_at_tile = random.choices(resource_types, weights=wtr_weights, k=1)[0]
        else:
            resource_at_tile = random.choices(resource_types, weights=weights, k=1)[0] 
        resource_ID = key = next((k for k, v in resources_dict.items() if v == resource_at_tile), 0) #0 acts as "None"
        if resource_ID is None:
            raise ValueError(f"Invalid resource name '{resource_at_tile}' (not in resources_dict)")

        resource_map[x][y] = resource_ID
        used_mask[x][y] = True


    

    resource_colors = [
    "#000000",  # 0: None (black)
    "#228B22",  # 1: Wood (green)
    "#f5deb3",  # 2: Salt (wheat)
    "#4B4B4B",  # 3: Coal (dark gray)
    "#A9A9A9",  # 4: Iron (gray)
    "#FFD700",  # 5: Gold (vibrant yellow)
    "#ADFF2F",  # 6: Grain (green-yellow / spring green)
    "#8B4513",  # 7: Oil (brown)
    "#808080",  # 8: Stone (stone gray)
]

    plt.figure("RESOURCE MAP")
    plt.imshow(resource_map, cmap=ListedColormap(resource_colors),interpolation='nearest')
    plt.title("RESOURCE MAP")
    plt.colorbar()

    return resources_dict, resource_map
# def populate_resources_2(altitude_map, fault_lines_map, size, temp_type, water_threshold, mtn_threshold, WORLD_SEED):
#     # 2 = converge
#     # -2 = diverge
#     # 1 = other faultlines

#     surface_resource_map = [[set() for _ in range(WORLD_SIZE)] for _ in range(WORLD_SIZE)]
#     subterranean_resource_map = [[set() for _ in range(WORLD_SIZE)] for _ in range(WORLD_SIZE)]

#     surface_resources_dict = {
#         0: "None",
#         1: "Wood",
#         2: "Grain",
#     }
#     subterranean_resources_dict = {
#         0: "None",
#         3: "Coal",
#         4: "Iron",
#         5: "Gold",
#         6: "Salt",
#         7: "Oil",
#         8: "Stone"
#     }
#     wtr_resources_dict = {
#         0: "None",
#         6: "Salt",
#         7: "Oil",
#     }
#     resources_rarity = {
#         "None" : 1,
#         "Stone" : 0.0,
#         "Wood" : 0.8,
#         "Salt" : 0,
#         "Coal" : 0.1,
#         "Iron" : 0.1,
#         "Gold" : 0.1,
#         "Grain" : 0.9,
#         "Oil" : 0.1
#     }
#     con_resources_rarity = {
#         "None" : .6,
#         "Stone" : 2,
#         "Wood" : 0.05,
#         "Salt" : 0,
#         "Coal" : 0.5,
#         "Iron" : 0.6,
#         "Gold" : 0.4,
#         "Grain" : 0,
#         "Oil" : 0.2
#     }
#     div_resources_rarity = {
#         "None" : 1,
#         "Stone" : 0.8,
#         "Wood" : 0.0,
#         "Salt" : 0.5,
#         "Coal" : 0.5,
#         "Iron" : 0.6,
#         "Gold" : 0.2,
#         "Grain" : 0,
#         "Oil" : 0.8
#     }
#     tran_resources_rarity = {
#         "None" : 1,
#         "Stone" : 1,
#         "Wood" : 2,
#         "Salt" : 0,
#         "Coal" : 0.7,
#         "Iron" : 0.6,
#         "Gold" : 0.3,
#         "Grain" : 3,
#         "Oil" : 0.6
#     }
#     wtr_resources_rarity = {
#         "None" : 2,
#         "Stone" : 0.0,
#         "Wood" : 0.0,
#         "Salt" : 0.9,
#         "Coal" : 0,
#         "Iron" : 0,
#         "Gold" : 0,
#         "Grain" : 0.0,
#         "Oil" : 0.6
#     }
#     resource_types = list(resources_rarity.keys())

#     random.seed(WORLD_SEED)

#     threshold = 10
#     convergent_mask = (fault_lines_map == 2).astype(float)
#     dist_div = distance_transform_edt(1 - convergent_mask)
#     divergent_mask = (fault_lines_map == -2).astype(float)
#     dist_conv = distance_transform_edt(1 - divergent_mask)
#     transform_mask = (fault_lines_map == 1).astype(float)
#     dist_tran = distance_transform_edt(1 - transform_mask)

#     weights = list(resources_rarity.values())   
#     con_weights = list(con_resources_rarity.values())   
#     div_weights = list(div_resources_rarity.values())  
#     tran_weights = list(tran_resources_rarity.values())  
#     wtr_weights = list(wtr_resources_rarity.values())  
#     used_mask = np.zeros((size, size), dtype=bool)

#     conv_indices = np.argwhere(dist_conv < threshold)
#     for y, x in conv_indices:
#         if altitude_map[x][y] <= water_threshold:
#             resource_at_tile = random.choices(resource_types, weights=wtr_weights, k=1)[0]
#         else:
#             resource_at_tile = random.choices(resource_types, weights=con_weights, k=1)[0]

#         resource_ID = next((k for k, v in resources_dict.items() if v == resource_at_tile), None)
#         if resource_ID is None:
#             raise ValueError(f"Invalid resource name '{resource_at_tile}' (not in resources_dict)")
#         resource_map[x][y] = resource_ID
#         used_mask[x][y] = True

#     div_indices = np.argwhere(dist_div < threshold)
#     for y, x in div_indices:
#         if altitude_map[x][y] < water_threshold:
#             resource_at_tile = random.choices(resource_types, weights=wtr_weights, k=1)[0]
#         else:
#             resource_at_tile = random.choices(resource_types, weights=div_weights, k=1)[0]
#         resource_ID = next((k for k, v in resources_dict.items() if v == resource_at_tile), None)
#         if resource_ID is None:
#             raise ValueError(f"Invalid resource name '{resource_at_tile}' (not in resources_dict)")
#         resource_map[x][y] = resource_ID
#         used_mask[x][y] = True

#     tran_indices = np.argwhere(dist_tran < threshold)
#     for y, x in tran_indices:
#         if altitude_map[x][y] < water_threshold:
#             resource_at_tile = random.choices(resource_types, weights=wtr_weights, k=1)[0]
#         else:
#             resource_at_tile = random.choices(resource_types, weights=tran_weights, k=1)[0]
#         resource_ID = next((k for k, v in resources_dict.items() if v == resource_at_tile), None)
#         if resource_ID is None:
#             raise ValueError(f"Invalid resource name '{resource_at_tile}' (not in resources_dict)")
#         resource_map[x][y] = resource_ID
#         used_mask[x][y] = True

#     other_indices = np.argwhere(used_mask == False)
#     for y, x in other_indices:
#         if altitude_map[x][y] < water_threshold:
#             resource_at_tile = random.choices(resource_types, weights=wtr_weights, k=1)[0]
#         else:
#             resource_at_tile = random.choices(resource_types, weights=weights, k=1)[0]
#         resource_ID = next((k for k, v in resources_dict.items() if v == resource_at_tile), None)
#         if resource_ID is None:
#             raise ValueError(f"Invalid resource name '{resource_at_tile}' (not in resources_dict)")
#         resource_map[x][y] = resource_ID
#         used_mask[x][y] = True

#     # Final validation
#     valid_ids = set(resources_dict.keys())
#     invalid_ids = np.unique(resource_map[~np.isin(resource_map, list(valid_ids))])
#     if len(invalid_ids) > 0:
#         raise ValueError(f"Invalid resource IDs in resource_map: {invalid_ids}")

#     resource_colors = [
#         "#000000",  # 0: None
#         "#228B22",  # 1: Wood
#         "#f5deb3",  # 2: Salt
#         "#4B4B4B",  # 3: Coal
#         "#A9A9A9",  # 4: Iron
#         "#FFD700",  # 5: Gold
#         "#ADFF2F",  # 6: Grain
#         "#8B4513",  # 7: Oil
#         "#808080",  # 8: Stone
#     ]

#     plt.figure("RESOURCE MAP")
#     plt.imshow(resource_map, cmap=ListedColormap(resource_colors), interpolation='nearest')
#     plt.title("RESOURCE MAP")
#     plt.colorbar()
#     plt.axis('off')
    
#     return resources_dict, resource_map

def populate_resources_3(altitude_map, fault_lines_map, size, temp_type, water_threshold, mtn_threshold, WORLD_SEED):
    # 2 = converge, -2 = diverge, 1 = transform

    surface_resource_map = np.zeros((size, size), dtype=int)
    subterranean_resource_map = np.zeros((size, size), dtype=int)

    surface_resources_dict = {
        0: "None",
        1: "Wood",
        2: "Grain",
    }
    subterranean_resources_dict = {
        0: "None",
        3: "Coal",
        4: "Iron",
        5: "Gold",
        6: "Salt",
        7: "Oil",
        8: "Stone"
    }
    wtr_resources_dict = {
        0: "None",
        6: "Salt",
        7: "Oil",
    }

    # Combine all resources for rarity weight references
    all_resources_dict = {**surface_resources_dict, **subterranean_resources_dict}
    reverse_resources_dict = {v: k for k, v in all_resources_dict.items()}

    resources_rarity = {
        "None": 1,
        "Stone": 0.0,
        "Wood": 0.8,
        "Salt": 0,
        "Coal": 0.1,
        "Iron": 0.1,
        "Gold": 0.1,
        "Grain": 0.9,
        "Oil": 0.1
    }
    con_resources_rarity = {
        "None": .6, "Stone": 2, "Wood": 0.05, "Salt": 0,
        "Coal": 0.5, "Iron": 0.6, "Gold": 0.4, "Grain": 0, "Oil": 0.2
    }
    div_resources_rarity = {
        "None": 1, "Stone": 0.8, "Wood": 0.0, "Salt": 0.5,
        "Coal": 0.5, "Iron": 0.6, "Gold": 0.2, "Grain": 0, "Oil": 0.8
    }
    tran_resources_rarity = {
        "None": 1, "Stone": 1, "Wood": 2, "Salt": 0,
        "Coal": 0.7, "Iron": 0.6, "Gold": 0.3, "Grain": 3, "Oil": 0.6
    }
    wtr_resources_rarity = {
        "None": 2, "Stone": 0.0, "Wood": 0.0, "Salt": 0.9,
        "Coal": 0, "Iron": 0, "Gold": 0, "Grain": 0.0, "Oil": 0.6
    }

    random.seed(WORLD_SEED)

    threshold = 10
    convergent_mask = (fault_lines_map == 2).astype(float)
    dist_div = distance_transform_edt(1 - convergent_mask)
    divergent_mask = (fault_lines_map == -2).astype(float)
    dist_conv = distance_transform_edt(1 - divergent_mask)
    transform_mask = (fault_lines_map == 1).astype(float)
    dist_tran = distance_transform_edt(1 - transform_mask)

    used_mask = np.zeros((size, size), dtype=bool)

    def assign_resources(x, y, rarity_map):
        if altitude_map[x][y] < water_threshold:
            sub_weights = [wtr_resources_rarity[r] for r in wtr_resources_dict.values()]
            sub_choice = random.choices(list(wtr_resources_dict.values()), weights=sub_weights, k=1)[0]
        else:
            sub_weights = [rarity_map[r] for r in subterranean_resources_dict.values()]
            sub_choice = random.choices(list(subterranean_resources_dict.values()), weights=sub_weights, k=1)[0]

        surf_weights = [rarity_map[r] for r in surface_resources_dict.values()]
        surf_choice = random.choices(list(surface_resources_dict.values()), weights=surf_weights, k=1)[0]

        surface_resource_map[x][y] = reverse_resources_dict[surf_choice]
        subterranean_resource_map[x][y] = reverse_resources_dict[sub_choice]
        used_mask[x][y] = True

    conv_indices = np.argwhere(dist_conv < threshold)
    for y, x in conv_indices:
        assign_resources(x, y, con_resources_rarity)

    div_indices = np.argwhere(dist_div < threshold)
    for y, x in div_indices:
        assign_resources(x, y, div_resources_rarity)

    tran_indices = np.argwhere(dist_tran < threshold)
    for y, x in tran_indices:
        assign_resources(x, y, tran_resources_rarity)

    other_indices = np.argwhere(~used_mask)
    for y, x in other_indices:
        assign_resources(x, y, resources_rarity)

    grain_id = 2  # assuming this is the correct ID
    grain_tiles = np.sum(surface_resource_map == grain_id)
    print(f"Grain tiles on surface: {grain_tiles}")
    
    resource_colors = [
    "#000000",  # 0: None
    "#228B22",  # 1: Wood
    "#ADFF2F",  # 2: Grain
    "#4B4B4B",  # 3: Coal
    "#A9A9A9",  # 4: Iron
    "#FFD700",  # 5: Gold
    "#f5deb3",  # 6: Salt
    "#8B4513",  # 7: Oil
    "#808080",  # 8: Stone
]
    cmap = ListedColormap(resource_colors)

    plt.figure("SURFACE RESOURCE MAP")
    plt.imshow(surface_resource_map, cmap=cmap, interpolation='nearest', vmin=0, vmax=8)
    plt.title("SURFACE RESOURCE MAP")
    plt.colorbar(ticks=range(len(resource_colors)))

    plt.figure("SUBTERRANEAN RESOURCE MAP")
    plt.imshow(subterranean_resource_map, cmap=cmap, interpolation='nearest', vmin=0, vmax=8)
    plt.title("SUBTERRANEAN RESOURCE MAP")
    plt.colorbar(ticks=range(len(resource_colors)))
    return surface_resource_map, subterranean_resource_map

                

# MAP GENERATION
def create_altitude_map(size, plate_list,plate_map,is_vor_border_map,fault_lines_map, WORLD_SEED): #create land noise
    zoom_level = 4.0 # you can tweak this higher/lower

    #lower at higher
    scale = zoom_level / size
    # rng = random.Random(WORLD_SEED)
    # scale = zoom_level / size

    #ASSIGN PLATES A BIAS AND RUGGEDNESS
    altitude_map = np.zeros((size,size))
   
    plate_type_mask = np.zeros((size, size)) #1 = ocean
   #BASE BlANKET OF NOISE
    row_offset = rng.uniform(0, 50)
    col_offset = rng.uniform(0, 50)
    for row in range(size):
        for col in range(size):
            

            nx = ((col)* scale + col_offset) 
            ny = ((row)* scale + row_offset) 
            pval = pnoise2 ( ny,nx,
                            octaves=4,
                            persistence=0.5,
                            lacunarity=2.0,
                            base=WORLD_SEED%256)
            altitude = (pval + 1)/2.0 #normalize btw [0,1]
            altitude_map[row][col] = altitude

    
    
    # FAULT EFFECTS
    adjusted_map = np.copy(altitude_map)
    falloff, magnitude = 5.0, 3.0

    convergent_mask = (fault_lines_map == 2).astype(float)
    dist_conv = distance_transform_edt(1 - convergent_mask)
    # Get current terrain base
    base = np.copy(altitude_map)

    # Faults raise elevation: blend fault height (1.0) with Perlin terrain
    conv_weight = np.exp(-dist_conv / 3)
    adjusted_map = base * (1 - conv_weight) + 1.0 * conv_weight

    divergent_mask = (fault_lines_map == -2).astype(float)
    dist_div = distance_transform_edt(1 - divergent_mask)
    adjusted_map -= np.exp(-dist_div / falloff) * magnitude

    adjusted_map = np.clip(adjusted_map, 0.0, 1.0)

    min_val = min(min(row) for row in altitude_map)
    max_val = max(max(row) for row in altitude_map)
    # print (f"MIN: {min_val} MAX: {max_val}")
    return adjusted_map
def Display_Interactive_Maps(altitude_map,temperature_map,temp_type,size,WORLD_SEED,SEED_AS_STRING):
    # COLOR DISPLAY
    temp = temp_type
    print(f"Requested colormap climate key: '{temp}'")

    bounds = [0.0, 0.4, 0.435, 0.6, 0.9, 1.0]
    colors = {}
    colors["cold"] = {
    "ocean": "#1B3B6F",           # Deep arctic blue
    "sand": "#E1D9D1",            # Pale icy shoreline
    "grass": "#A0C4B0",           # Cold tundra green
    "lower_mountain": "#8B9DA6",  # Frosted slate gray
    "peak": "#FFFFFF"             # Snowy peak white
    }   
    colors["mild"] = {
    "ocean": "#2E8BC0",           # Calm ocean blue
    "sand": "#F2E2C4",            # Warm beige sand
    "grass": "#7BC47F",           # Lush green plains
    "lower_mountain": "#A9A9A9",  # Granite gray
    "peak": "#FFFFFF"             # Snow-capped white
}
    colors["hot"] = {
        "ocean": "#005C5C",           # Warm tropical sea
        "sand": "#E0B084",            # Desert sand
        "grass": "#C2B280",           # Dry savanna grass
        "lower_mountain": "#A0522D",  # Reddish-brown rock
        "peak": "#FFFAF0"             # Sun-bleached summit
    }

    # colors = [
    #     '#1f4e79',  # Deep water
    #     '#e0c074',  # Sand
    #     '#5cb85c',  # Grass
    #     '#888888',  # Mountains
    #     '#ffffff'   # Peaks
    # ]
    terrain_cmap = ListedColormap([
    colors[temp]["ocean"],
    colors[temp]["sand"],
    colors[temp]["grass"],
    colors[temp]["lower_mountain"],
    colors[temp]["peak"]
])

    terrain_norm = BoundaryNorm(bounds, terrain_cmap.N)
    
    colors = [
        "#0000cc",   # Blue for < 0
        "#C2B280",   # tan
        "#228B22",   # Green
        "#707070",   # Gray
        "#ffffff"    # White
    ]

    # Create colormap and norm
    cmap_altitude = ListedColormap(colors)
    fig1, ax1 = plt.subplots()
    img1 = ax1.imshow(altitude_map, cmap=cmap_altitude, norm=terrain_norm)
    cbar1 = plt.colorbar(img1, boundaries=bounds)
    plt.axis('off')
    cbar1.set_label("Elevation")
    zoom_radius1 = 30
    center_x1, center_y1 = size // 2, size // 2

    def update_view_1():
        ax1.set_xlim(center_x1 - zoom_radius1, center_x1 + zoom_radius1)
        ax1.set_ylim(center_y1 + zoom_radius1, center_y1 - zoom_radius1)
        fig1.canvas.draw_idle()

    def on_key_1(event):
        nonlocal center_x1, center_y1, zoom_radius1
        step = 5
        zoom_step = 5
        if event.key == 'left': center_x1 -= step
        elif event.key == 'right': center_x1 += step
        elif event.key == 'up': center_y1 -= step
        elif event.key == 'down': center_y1 += step
        elif event.key in ('+', '='): zoom_radius1 = max(5, zoom_radius1 - zoom_step)
        elif event.key == '-': zoom_radius1 = min(size // 2, zoom_radius1 + zoom_step)
        center_x1 = max(zoom_radius1, min(center_x1, size - zoom_radius1))
        center_y1 = max(zoom_radius1, min(center_y1, size - zoom_radius1))
        update_view_1()

    fig1.canvas.mpl_connect('key_press_event', on_key_1)
    ax1.set_title(f"[ELEVATION] World: {SEED_AS_STRING}")
    update_view_1()


    fig2, ax2 = plt.subplots()
    img2 = ax2.imshow(altitude_map, cmap=terrain_cmap, norm=terrain_norm)
    cbar2 = plt.colorbar(img2, boundaries=bounds)
    plt.axis('off')
    cbar2.set_label("Temperature")
    zoom_radius2 = 30
    center_x2, center_y2 = size // 2, size // 2

    def update_view_2():
        ax2.set_xlim(center_x2 - zoom_radius2, center_x2 + zoom_radius2)
        ax2.set_ylim(center_y2 + zoom_radius2, center_y2 - zoom_radius2)
        fig2.canvas.draw_idle()

    def on_key_2(event):
        nonlocal center_x2, center_y2, zoom_radius2
        step = 5
        zoom_step = 5
        if event.key == 'left': center_x2 -= step
        elif event.key == 'right': center_x2 += step
        elif event.key == 'up': center_y2 -= step
        elif event.key == 'down': center_y2 += step
        elif event.key in ('+', '='): zoom_radius2 = max(5, zoom_radius2 - zoom_step)
        elif event.key == '-': zoom_radius2 = min(size // 2, zoom_radius2 + zoom_step)
        center_x2 = max(zoom_radius2, min(center_x2, size - zoom_radius2))
        center_y2 = max(zoom_radius2, min(center_y2, size - zoom_radius2))
        update_view_2()

    fig2.canvas.mpl_connect('key_press_event', on_key_2)
    ax2.set_title(f"[TEMPERATURE] World: {SEED_AS_STRING}")
    update_view_2()

    # Raise windows
    try:
        fig1.canvas.manager.window.activateWindow()
        fig1.canvas.manager.window.raise_()
        fig2.canvas.manager.window.activateWindow()
        fig2.canvas.manager.window.raise_()
    except:
        pass
def create_temp_map(size,WORLD_SEED):
    
    # rng = random.Random(WORLD_SEED)
    val = rng.uniform(0,101)
    temp = ""
    base_val = 0
    if val < 33: 
        temp = "cold"
    elif val < 66: 
        temp = "mild"

    else: 
        temp = "hot"
    max_temp = val
    temp_map = np.zeros((size,size))
    print(temp)
    #NOISE GENERATION
    zoom_level = 4.0 # you can tweak this higher/lower
    #lower at higher
    scale = zoom_level / size
    row_offset = rng.uniform(0, 50)
    col_offset = rng.uniform(0, 50)
    for row in range(size):
        for col in range(size):
            distance = np.sqrt(row**2 + col**2)
            max_distance = np.sqrt((size-1)**2 + (size-1)**2)
            normalized_distance = distance / max_distance

            temperature = max_temp * (1 - normalized_distance)

            temp_map[row][col] = temperature
    add_perlin_noise(temp_map,scale,5,WORLD_SEED)
    temp_map = np.clip(temp_map, 0, None)  # Remove negatives
    temp_map = (temp_map - np.min(temp_map)) / (np.max(temp_map) - np.min(temp_map))

    # plt.figure("Temperature")
    # plt.imshow(temp_map,"gray")

    return temp, temp_map
def add_perlin_noise(temp_map, scale=0.1, amplitude=5, seed=0): # CHAT GPT'D
    height, width = temp_map.shape
    noisy_temp = np.copy(temp_map)
    for row in range(height):
        for col in range(width):
            nx = row * scale
            ny = col * scale
            noise_val = pnoise2(nx, ny, octaves=4, base = seed % (2**31))  # safe for pnoise2)
            noisy_temp[row, col] += amplitude * noise_val
    return noisy_temp

#CIV
def find_possible_civ_origins(surface_resource_map,altitude_map,temperature_map,size,civ_land_map):
    resources_dict = {
        0: "None",
        1: "Wood",
        2: "Grain",
        3: "Coal",
        4: "Iron",
        5: "Gold",
        6: "Salt",
        7: "Oil",
        8: "Stone"
    }
    possible_settlements = []
    valid = []
    grain_total = 0
    grain_pass_alt = 0
    grain_pass_all = 0

    for y in range(size):
        for x in range(size):
            if resources_dict[surface_resource_map[y][x]] == "Grain":
                grain_total += 1
                if 0.4 < altitude_map[y][x] < 0.8:
                    grain_pass_alt += 1
                    if civ_land_map[y][x] == 0:
                        grain_pass_all += 1
                        possible_settlements.append(tuple((y,x)))

    print(f"Total Grain Tiles: {grain_total}")
    print(f"  ↳ Pass Altitude Range: {grain_pass_alt}")
    print(f"  ↳ Pass Altitude + Not Claimed: {grain_pass_all}")

    print(f"Valid possible civ origins: {len(valid)}")

    return possible_settlements



    

# seedAsString = input("Enter a World Seed: ")

WORLD_SIZE = 100

print("Starting Program...")
    
#SEED GENERATION
seedAsString = create_seed()
WORLD_SEED = seed_from_string(seedAsString)
random.seed(WORLD_SEED)
rng = np.random.default_rng(WORLD_SEED)
print(f"Seed: {seedAsString} ({WORLD_SEED})")


prefixes = [
    "Ael", "Thal", "Vor", "Kar", "Zan", "My", "Eri", "Gor", "Ul", "Ser",
    "Bel", "Dra", "Mal", "Fen", "Tor", "Bar", "Ash", "Caer", "Vul", "Nor",
    "El", "Rav", "Thorn", "Gal", "Syl", "Dur", "Nym", "Mor", "Lun", "Isen",
    "Ark", "Cal", "Faer", "Or", "Aer", "Grim", "Hal", "Jor", "Val", "Tir",
    "Ebon", "Bran", "Xan", "Yl", "Saer", "Teth", "Zor", "Quel", "Naer", "Kyr"
]

suffixes = [
    "dor", "mere", "heim", "grad", "wyn", "moor", "hollow", "spire", "gate", "reach",
    "hold", "keep", "stead", "crest", "haven", "cliff", "watch", "run", "forge", "deep",
    "grove", "bluff", "shade", "pass", "fall", "peak", "den", "mark", "rock", "bastion",
    "fell", "vale", "thorne", "flame", "storm", "crag", "rift", "glade", "mount", "shard",
    "bay", "cairn", "point", "river", "meadow", "fen", "brook", "chasm", "barrow", "vault"
]





class Civilization:
    def print_summary(self):
        print("="*40)
        print(f"Summary for Civilization: {self.name}")
        print(f"Age: {self.age} turns")
        print(f"Number of Cities: {len(self.cities)}")
        
        total_population = sum(city.population for city in self.cities)
        print(f"Total Population: {total_population}")

        print("Resources:")
        for resource, amount in self.resources.items():
            print(f"  {resource}: {amount}")

        print("City Details:")
        for i, city in enumerate(self.cities):
            print(f"  {i+1}. {city.name}")
            print(f"     Population: {city.population}")
            if hasattr(city, "location"):
                print(f"     Location: {city.location}")
            if hasattr(city, "buildings") and city.buildings:
                if isinstance(city.buildings, dict):
                    building_list = list(city.buildings.keys())
                else:
                    building_list = list(city.buildings)
                print(f"     Buildings: {', '.join(building_list)}")
            else:
                print(f"     Buildings: None")
        print("="*40)
    @staticmethod
    def generate_fantasy_nation():
        nation_prefixes = ['Kingdom of', 'Empire of', 'The Republic of', 'Dominion of', 'The Free Lands of']
        return f"{random.choice(nation_prefixes)} {rng.choice(prefixes) + rng.choice(suffixes)}"
    def __init__(self, origin_coords):
        self.name = self.generate_fantasy_nation()
        self.tiles = set([tuple(origin_coords)])
        self.population = 50
        self.resources = {
            "None": 0,
            "Stone": 100,
            "Wood": 100,
            "Salt": 0,
            "Coal": 0,
            "Iron": 0,
            "Gold": 0,
            "Grain": 100,
            "Oil": 0
        }
        self.id = origin_coords
        self.age = 0
        self.tech_level = 0
        self.neighbors = {}

        self.cities = list([City(self.name, "Capital", origin_coords)])

        self.resources_dict = {
        0: "None",
        1: "Wood",
        2: "Grain",
        3: "Coal",
        4: "Iron",
        5: "Gold",
        6: "Salt",
        7: "Oil",
        8: "Stone"
    }

    def settle_city(self, resource_map, altitude_map, civ_land_map):
        possible_settlements = []
        tiles_of_interest = set()
        for cities in self.cities:
            surrounding_tiles = cities.get_surrounding_tiles(cities.location, WORLD_SIZE, 10)
            tiles_of_interest.update(surrounding_tiles)
        for y, x in tiles_of_interest:
            if (self.resources_dict[resource_map[0][y][x]] == "Grain" and
                0.4 < altitude_map[y][x] < 0.8 and civ_land_map[y][x] == 0):
                possible_settlements.append((y, x))
        if not possible_settlements:
            return False
        coords = rng.choice(possible_settlements)
        number = len(self.cities) + 1
        city_name = "City " + str(number)

        new_city = City(self.name, city_name, coords)
        self.cities.append(new_city)
        self.tiles.update(new_city.tiles)
        return True

    @staticmethod
    def can_afford(nation_resources, cost_dict):
        return all(nation_resources.get(res, 0) >= cost for res, cost in cost_dict.items())

    def can_afford_city_building(self, city):
        current_upgrade = len(city.buildings)
        if current_upgrade == 0:
            return self.can_afford(self.resources, city.city_upgrades["Granary"]["cost"])
        elif current_upgrade == 1:
            return self.can_afford(self.resources, city.city_upgrades["Workshop"]["cost"])
        elif current_upgrade == 2:
            return self.can_afford(self.resources, city.city_upgrades["Marketplace"]["cost"])
        else:
            return False

    def take_action(self, resource_map, altitude_map, civ_land_map):
        possible_actions = []
        settlement_cost = {
            "Grain": 100,
            "Wood": 60,
            "Stone": 40,
        }
        if self.can_afford(self.resources, settlement_cost):
            possible_actions.append("settle_city")
        for index, city in enumerate(self.cities):
            if self.can_afford_city_building(city):
                possible_actions.append({index: "upgrade_city"})

        action = rng.choice(possible_actions) if possible_actions else None

        if action == "settle_city":
            success = self.settle_city(resource_map, altitude_map, civ_land_map)
            if success:
                for res, cost in settlement_cost.items():
                    self.resources[res] -= cost
        elif isinstance(action, dict):
            key, val = next(iter(action.items()))
            if val == "upgrade_city":
                self.cities[key].upgrade_city(self.resources)

    def simulate_turn(self, surface_resource_map, subt_resource_map, altitude_map, civ_land_map):
        resource_map = np.array([surface_resource_map, subt_resource_map])
        curr_pop = 0
        for city in self.cities:
            city.simulate_city_turn(self.resources, self.tiles, resource_map)
            curr_pop += city.population
        self.population = curr_pop
        self.take_action(resource_map, altitude_map, civ_land_map)
        self.age += 1


class City:
    @staticmethod
    def generate_city_name():
        return rng.choice(prefixes) + rng.choice(suffixes)
    def __init__(self, nation, name, location_coords):
        self.owner = nation
        self.name = self.generate_city_name()
        self.location = location_coords
        self.tiles = City.get_surrounding_tiles(location_coords, WORLD_SIZE, 1)
        self.population = 20
        self.age = 0
        self.current_radius = 1
        self.radius_threshold = {
            1: 0, 2: 40, 3: 80, 4: 160, 5: 250,
            6: 500, 7: 800, 8: 1200, 9: 1600, 10: 2500
        }
        self.buildings = []
        self.next_upgrade = 0
        self.city_upgrades = {
            "Granary": {
                "description": "Improves grain production capacity.",
                "cost": {"Wood": 50, "Stone": 30},
                "benefits": {"Grain": 10},
                "requires": None
            },
            "Workshop": {
                "description": "Increases production of refined materials.",
                "cost": {"Wood": 80, "Stone": 50, "Gold": 20},
                "benefits": {"Iron": 5, "Coal": 5},
                "requires": "Granary"
            },
            "Marketplace": {
                "description": "Boosts gold income through trade.",
                "cost": {"Wood": 60, "Stone": 40, "Gold": 50},
                "benefits": {"Gold": 25},
                "requires": "Workshop"
            },
        }
        self.resources_dict = {
        0: "None",
        1: "Wood",
        2: "Grain",
        3: "Coal",
        4: "Iron",
        5: "Gold",
        6: "Salt",
        7: "Oil",
        8: "Stone"
    }

    def city_gain(self, resource_map):
        resources_gathered = defaultdict(int)
        resources_gathered["Grain"] += 3
        resources_gathered["Stone"] += 3
        resources_gathered["Wood"] += 3
        for y, x in self.tiles:
            if 0 <= y < WORLD_SIZE and 0 <= x < WORLD_SIZE:
                for layer in range(2):
                    resource = self.resources_dict[resource_map[layer][y][x]]
                    if resource != "None":
                        resources_gathered[resource] += 1
            resources_gathered["Gold"] += self.population * 0.2
        return resources_gathered

    def city_upkeep(self):
        grain_required = self.population // 20
        wood_required = self.population // 50
        stone_required = self.population // 100
        gold_required = 0
        return {
            "Grain": grain_required,
            "Wood": wood_required,
            "Stone": stone_required,
            "Gold": gold_required
        }

    def city_maint(self, civ_resources, resource_map):
        resources_gained = self.city_gain(resource_map)
        resources_spent = self.city_upkeep()
        for resource in self.resources_dict.values():
            if resource == "None":
                continue
            net_change = resources_gained[resource] - resources_spent.get(resource, 0)
            new_tot = civ_resources[resource] + net_change
            if new_tot < 0:
                civ_resources[resource] = 0
                if resource == "Grain":
                    self.population *= 0.8
                elif resource == "Gold":
                    self.population *= 0.9
            else:
                civ_resources[resource] = new_tot
                if resource == "Grain":
                    self.population += net_change * 1.0

    def simulate_city_turn(self, civ_resources, civ_tiles, resource_map):
        self.city_maint(civ_resources, resource_map)
        if self.population > self.radius_threshold.get(min(self.current_radius + 1, 10), float('inf')):
            self.current_radius += 1
            new_tiles = self.get_surrounding_tiles(self.location, WORLD_SIZE, self.current_radius)
            self.tiles.update(new_tiles)
            civ_tiles.update(self.tiles)
        self.age += 1

    @staticmethod
    def get_surrounding_tiles(origin, WORLD_SIZE, radius):
        yc, xc = tuple(origin)
        height, width = WORLD_SIZE, WORLD_SIZE
        tiles = set()
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = yc + dy, xc + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if np.sqrt(dy**2 + dx**2) <= radius:
                        tiles.add((ny, nx))
        return tiles

    @staticmethod
    def can_afford(nation_resources, cost_dict):
        return all(nation_resources.get(res, 0) >= cost for res, cost in cost_dict.items())

    def upgrade_city(self, civ_resources):
        current_upgrade = len(self.buildings)
        if current_upgrade == 0:
            upgrade = "Granary"
        elif current_upgrade == 1:
            upgrade = "Workshop"
        elif current_upgrade == 2:
            upgrade = "Marketplace"
        else:
            return
        if self.can_afford(civ_resources, self.city_upgrades[upgrade]["cost"]):
            for k, v in self.city_upgrades[upgrade]["cost"].items():
                civ_resources[k] -= v
            self.buildings.append(upgrade)


def update_civ_map(civ_map,civilizations):
    for index,civ in enumerate(civilizations):
        for y,x in civ.tiles:
            civ_map[y][x] = (index + 1)


    

def main():
    
    
    #PLATE GENERATION
    seeds,vor_regions = Voronoi_seeding(WORLD_SIZE,0.00010,WORLD_SEED)
    plates = create_tectonic_plates(seeds,vor_regions,WORLD_SIZE,WORLD_SEED)
    is_vor_border,fault_lines = create_fault_lines(WORLD_SIZE,vor_regions,plates,seeds)
    #MAP CREATION
    altitude = create_altitude_map(WORLD_SIZE,plates,vor_regions,is_vor_border,fault_lines,WORLD_SEED)
    temp_type, temperature = create_temp_map(WORLD_SIZE,WORLD_SEED)
    surface_resources,subt_resources = populate_resources_3(altitude,fault_lines,WORLD_SIZE,temp_type,0.4,0.6,WORLD_SEED)

    # print("Unique values in surface_resources after generation:", np.unique(surface_resources))
    # print("Unique values in subt_resources after generation:", np.unique(subt_resources))

    
    

    #DISPlAY MAPS
    Display_Interactive_Maps(altitude,temperature,temp_type,WORLD_SIZE,WORLD_SEED,seedAsString)
    # plt.show()


    #CIVILIZATIONS
    fake = Faker()

    

    civilizations = []
    civ_territories_map = np.zeros((WORLD_SIZE,WORLD_SIZE))
    for civs in range(int(rng.uniform(1,6))): # 1 starting civilizations
        origin = tuple(rng.choice(find_possible_civ_origins(surface_resources,altitude,temperature,WORLD_SIZE,civ_territories_map)) )
        civilizations.append(Civilization(origin))


    update_civ_map(civ_territories_map,civilizations) #initial territories

    year = 0

    while year < 1000:
        update_civ_map(civ_territories_map,civilizations) #initial territories
        for civ in civilizations:
            civ.simulate_turn(surface_resources,subt_resources,altitude,civ_territories_map)
        year+=1
    
    for civs in civilizations:
        civs.print_summary()
   
    update_civ_map(civ_territories_map, civilizations)
    plt.figure()
    plt.imshow(civ_territories_map, cmap='tab20', interpolation='nearest', aspect='equal')
    plt.title("Civilization Territories")
    plt.axis('off')
    plt.show()

    






    return


print("RUNNING PROGRAM...")
main()
print("PROGRAM FINISHING...")
    

    


