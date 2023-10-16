from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym
import cv2 as cv
import numpy as np
import string

# code for locating objects on the screen in super mario bros
# by Lauren Gee

#edits made to this code will be marked with additional comments

# Template matching is based on this tutorial:
# https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

################################################################################

# change these values if you want more/less printing
PRINT_LOCATIONS = False

# If printing the grid doesn't display in an understandable way, change the
# settings of your terminal (or anaconda prompt) to have a smaller font size,
# so that everything fits on the screen. Also, use a large terminal window /
# whole screen.

# other constants (don't change these)
SCREEN_HEIGHT   = 240
SCREEN_WIDTH    = 256
MATCH_THRESHOLD = 0.9

################################################################################
# TEMPLATES FOR LOCATING OBJECTS

# ignore sky blue colour when matching templates
MASK_COLOUR = np.array([252, 136, 104]) #ground

# (these numbers are [BLUE, GREEN, RED] because opencv uses BGR colour format by default)

# You can add more images to improve the object locator, so that it can locate
# more things. For best results, paint around the object with the exact shade of
# blue as the sky colour. (see the given images as examples)
#
# Put your image filenames in image_files below, following the same format, and
# it should work fine.

# filenames for object templates
#addition: many more enemies, such as paratroopas, and blocks, such as bridges
image_files = {
    "mario": {
        "small": ["marioA.png", "marioB.png", "marioC.png", "marioD.png",
                  "marioE.png", "marioF.png", "marioG.png"],
        "tall": ["tall_marioA.png", "tall_marioB.png", "tall_marioC.png"],
    },
    "enemy": {
        "goomba": ["goomba.png"],
        "koopa": ["koopaA.png", "koopaB.png"],
        "paratroopa": ["paratroopaA.png", "paratroopaB.png"]
    },
    "hard_enemy": {
        "piranha plant": ["plantA.png", "plantB.png"]
    },
    "block": {
        "block": ["block1.png", "block2.png", "block3.png", "block4.png", "semisolid_left.png", "semisolid_middle.png", "semisolid_right.png", "bridge.png", "platform.png", "cloud.png", "cloud_platform.png"],
        "question_block": ["questionA.png", "questionB.png", "questionC.png"],
        "pipe": ["pipe_upper_section.png", "pipe_lower_section.png"],
    },
    "item": {
        # Note: The template matcher is colourblind (it's using greyscale),
        # so it can't tell the difference between red and green mushrooms.
        "mushroom": ["mushroom_red.png"],
        "spring": ["spring.png"],
        "vine": ["vine.png", "vine_top.png"]
        # There are also other items in the game that I haven't included,
        # such as star.

        # There's probably a way to change the matching to work with colour,
        # but that would slow things down considerably. Also, given that the
        # red and green mushroom sprites are so similar, it might think they're
        # the same even if there is colour.
    }
}



def _get_template(filename):
    image = cv.imread(filename)
    assert image is not None, f"File {filename} does not exist."
    template = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = np.uint8(np.where(np.all(image == MASK_COLOUR, axis=2), 0, 1))
    num_pixels = image.shape[0]*image.shape[1]
    if num_pixels - np.sum(mask) < 10:
        mask = None # this is important for avoiding a problem where some things match everything
    dimensions = tuple(template.shape[::-1])
    return template, mask, dimensions

def get_template(filenames):
    results = []
    for filename in filenames:
        results.append(_get_template(filename))
    return results

def get_template_and_flipped(filenames):
    results = []
    for filename in filenames:
        template, mask, dimensions = _get_template(filename)
        results.append((template, mask, dimensions))
        results.append((cv.flip(template, 1), cv.flip(mask, 1), dimensions))
    return results

# Mario and enemies can face both right and left, so I'll also include
# horizontally flipped versions of those templates.
include_flipped = {"mario", "enemy", "hard_enemy"}

# generate all templatees
templates = {}
for category in image_files:
    category_items = image_files[category]
    category_templates = {}
    for object_name in category_items:
        filenames = category_items[object_name]
        if category in include_flipped or object_name in include_flipped:
            category_templates[object_name] = get_template_and_flipped(filenames)
        else:
            category_templates[object_name] = get_template(filenames)
    templates[category] = category_templates

#the grid printing functions originally in Lauren's code were not used in our project (even in debugging), so they have been removed

################################################################################
# LOCATING OBJECTS

def _locate_object(screen, templates, stop_early=False, threshold=MATCH_THRESHOLD):
    locations = {}
    for template, mask, dimensions in templates:
        results = cv.matchTemplate(screen, template, cv.TM_CCOEFF_NORMED, mask=mask)
        locs = np.where(results >= threshold)
        for y, x in zip(*locs):
            locations[(x, y)] = dimensions

        # stop early if you found mario (don't need to look for other animation frames of mario)
        if stop_early and locations:
            break
    
    #      [((x,y), (width,height))]
    return [( loc,  locations[loc]) for loc in locations]

def _locate_pipe(screen, threshold=MATCH_THRESHOLD):
    upper_template, upper_mask, upper_dimensions = templates["block"]["pipe"][0]
    lower_template, lower_mask, lower_dimensions = templates["block"]["pipe"][1]

    # find the upper part of the pipe
    upper_results = cv.matchTemplate(screen, upper_template, cv.TM_CCOEFF_NORMED, mask=upper_mask)
    upper_locs = list(zip(*np.where(upper_results >= threshold)))
    
    # stop early if there are no pipes
    if not upper_locs:
        return []
    
    # find the lower part of the pipe
    lower_results = cv.matchTemplate(screen, lower_template, cv.TM_CCOEFF_NORMED, mask=lower_mask)
    lower_locs = set(zip(*np.where(lower_results >= threshold)))

    # put the pieces together
    upper_width, upper_height = upper_dimensions
    lower_width, lower_height = lower_dimensions
    locations = []
    for y, x in upper_locs:
        for h in range(upper_height, SCREEN_HEIGHT, lower_height):
            if (y+h, x+2) not in lower_locs:
                locations.append(((x, y), (upper_width, h), "pipe"))
                break
    return locations

def locate_objects(screen, mario_status):
    # convert to greyscale
    screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)

    # iterate through our templates data structure
    object_locations = {}
    for category in templates:
        category_templates = templates[category]
        category_items = []
        stop_early = False
        for object_name in category_templates:
            # use mario_status to determine which type of mario to look for
            if category == "mario":
                if object_name != mario_status:
                    continue
                else:
                    stop_early = True
            # pipe has special logic, so skip it for now
            if object_name == "pipe":
                continue
            
            # find locations of objects
            results = _locate_object(screen, category_templates[object_name], stop_early)
            for location, dimensions in results:
                category_items.append((location, dimensions, object_name))

        object_locations[category] = category_items

    # locate pipes
    object_locations["block"] += _locate_pipe(screen)

    return object_locations

################################################################################
# GETTING INFORMATION AND CHOOSING AN ACTION

def make_action(screen, info, step, env, prev_action):
    mario_status = info["status"]
    object_locations = locate_objects(screen, mario_status)

    # List of locations of Mario:
    mario_locations = object_locations["mario"]
    # (There's usually 1 item in mario_locations, but there could be 0 if we
    # couldn't find Mario. There might even be more than one item in the list,
    # but if that happens they are probably approximately the same location.)

    # List of locations of enemies, such as goombas and koopas:
    enemy_locations = object_locations["enemy"]

    # List of locations of enemies that can't be jumped on, such as piranha plants:
    hard_enemy_locations = object_locations["hard_enemy"]

    # List of locations of blocks, pipes, etc:
    block_locations = object_locations["block"]

    # List of locations of items: (so far, it only finds mushrooms)
    item_locations = object_locations["item"]

    # This is the format of the lists of locations:
    # ((x_coordinate, y_coordinate), (object_width, object_height), object_name)
    #
    # x_coordinate and y_coordinate are the top left corner of the object
    #
    # For example, the enemy_locations list might look like this:
    # [((161, 193), (16, 16), 'goomba'), ((175, 193), (16, 16), 'goomba')]
    #addition: defaults for mario's location based on where he's most likely located on the screen, for if the agent can't find him for whatever reason
    mario_x = 120
    mario_y = 79
    #edit: check mario's location and set the variables no matter what, since they're used in the decision making code
    if mario_locations:
        location, dimensions, object_name = mario_locations[0]
        mario_x, mario_y = location
        #addition: avoid breaking by adjusting Mario's coordinates if he's big, since the locating code measures from the top-right corner
        if info["status"] != 'small':
            mario_y -= 16
    if PRINT_LOCATIONS:
        # To get the information out of a list:
        for enemy in enemy_locations:
            enemy_location, enemy_dimensions, enemy_name = enemy
            x, y = enemy_location
            width, height = enemy_dimensions
            print("enemy:", x, y, width, height, enemy_name)

        for enemy in hard_enemy_locations:
            enemy_location, enemy_dimensions, enemy_name = enemy
            x, y = enemy_location
            width, height = enemy_dimensions
            print("hard enemy:", x, y, width, height, enemy_name)

        # Or you could do it this way:
        for block in block_locations:
            block_x = block[0][0]
            block_y = block[0][1]
            block_width = block[1][0]
            block_height = block[1][1]
            block_name = block[2]
            print(f"{block_name}: {(block_x, block_y)}), {(block_width, block_height)}")
        # Or you could do it this way:
        for item_location, item_dimensions, item_name in item_locations:
            x, y = item_location
            print(item_name, x, y)

        # gym-super-mario-bros also gives us some info that might be useful
        print(info)
        # see https://pypi.org/project/gym-super-mario-bros/ for explanations

        # The x and y coordinates in info are world coordinates.
        # They tell you where Mario is in the game, not his screen position.
        mario_world_x = info["x_pos"]
        mario_world_y = info["y_pos"]
        # Also, you can get Mario's status (small, tall, fireball) from info too.
        mario_status = info["status"]
        print("Mario's location in world:",
              mario_world_x, mario_world_y, f"({mario_status} mario)")
    #addition: all decision making code below
    #choose an action based on collected information
    hole = True
    grounded = False
    #check block locations to see if Mario is on the ground and if the platform he's on is ending
    for b in block_locations:
        if b[0][0] - mario_x in range(0, 20) and b[0][1] - mario_y in range(-20, 20):
            hole = False
        if b[0][0] - mario_x in range(-8, 8) and b[0][1] - mario_y in range(14, 18):
            grounded = True
    #see if there's something to jump over, assuming you're grounded and can jump
    if grounded:
        #print('grounded')
        if hole:
            #jump when you're at the edge of a platform
            '''mass printing freezes the screen for debugging purposes
            for i in range(250000):
                print("Found a pit, jumping!")
                print(mario_x, mario_y)
            '''
            return 4
        #print(enemy_locations)
        for e in enemy_locations:
            if e[0][0] - mario_x in range(1, 70) and e[0][1] - mario_y in range(-20, 20):
                #jump over nearby enemies
                '''mass printing freezes the screen for debugging purposes
                for i in range(250000):
                    print("Found an enemy, jumping!")
                    print("Mario coordinates:", mario_x, mario_y)
                '''
                return 4
            if e[0][0] - mario_x in range(-1, -70) and e[0][1] - mario_y in range(-20, 20):
                #jump over enemies coming from behind
                '''mass printing freezes the screen for debugging purposes
                for i in range(250000):
                    print("Found an enemy behind you, jumping!")
                    print("Mario coordinates:", mario_x, mario_y)
                '''
                return 9
        #print(hard_enemy_locations)
        for e in hard_enemy_locations:
            if e[0][0] - mario_x in range(1, 70) and e[0][1] - mario_y in range(-20, 20):
                #jump over nearby enemies
                '''mass printing freezes the screen for debugging purposes
                for i in range(250000):
                    print("Found an unstompable enemy, jumping!")
                    print("Mario coordinates:", mario_x, mario_y)
                '''
                return 4
            if e[0][0] - mario_x in range(-1, -70) and e[0][1] - mario_y in range(-20, 20):
                #jump over enemies coming from behind
                '''mass printing freezes the screen for debugging purposes
                for i in range(250000):
                    print("Found an unstompable enemy behind you, jumping!")
                    print("Mario coordinates:", mario_x, mario_y)
                '''
                return 9
        #space for more jumping responses
   
    #space for more non-jumping responses
    #by default run right
    return 3
################################################################################
#end of Lauren's code

#run from 1-1 with 3 lives
env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
#run from level of choice with 1 life
#env = gym.make("SuperMarioBros-1-3-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, COMPLEX_MOVEMENT)

obs, done = None, True
env.reset()
jumpCount, maxDist, blockedCount, triedSmall = 0, 0, 0, False
lives = 3
stage = (1,1)
rewardSum, steps = 0,0
for step in range(100000):
    if jumpCount > 0:
        #when jumping, keep holding jump to ensure a large jump is made
        jumpCount -= 1
        #print("JUMP!JUMP!JUMP!JUMP!JUMP!JUMP!JUMP!JUMP!" + str(jumpCount))
        #the last few frames of the jump release the jump button to ensure the button isn't kept held down, which would prevent consecutive jumps
        if jumpCount > 10:
            action = 4
        else:
            action = 3
    elif jumpCount < 0:
        #same as above for leftward jumps
        jumpCount += 1
        #print("JUMP?JUMP?JUMP?JUMP?JUMP?JUMP?JUMP?JUMP?" + str(jumpCount))
        if jumpCount < -10:
            action = 9
        else:
            action = 8
    #if Mario hasn't moved right in a while, try jumping to clear blocks/pipes
    elif blockedCount > 15 and not triedSmall:
        #start with a small jump to ascend staircases
        '''
        for i in range(100000):
            print("Blocked; jumping at short height...")
        '''
        action == 4
        jumpCount = 15
        blockedCount = 0
        triedSmall = True
    elif blockedCount > 35:
        #do a full sized jump if the small one didn't work
        '''
        for i in range(100000):
            print("Blocked; jumping at full height...")
        '''
        action == 4
        jumpCount = 35
        blockedCount = 0
    elif obs is not None:
        action = make_action(obs, info, step, env, action)
        #if you begin to jump, set the jumpCount variables accordingly
        if action == 4 and jumpCount == 0:
            jumpCount = 35
        elif action == 9 and jumpCount == 0:
            jumpCount = -35
    #run right as the first action + if the observation ever fails to be obtained
    else:
        action = 3
    obs, reward, terminated, truncated, info = env.step(action)
    '''Debug print statements to display notable information to the terminal
    print(action)
    print("Action performed: " + str(COMPLEX_MOVEMENT[action]))
    print(maxDist, blockedCount)
    print('Reward: ' + str(reward))
    '''
    rewardSum += reward
    steps += 1
    #check if Mario is still successfully moving right and start counting if he isn't
    if info["x_pos"] > maxDist:
        maxDist = info['x_pos']
        blockedCount = 0
        triedSmall = False
    #reset max distance on death or stage clear
    elif info["life"] < lives or (info["world"], info["stage"]) != stage:
        maxDist = 0
        lives = info['life']
        stage = (info["world"], info["stage"])
        #print("Current reward gained: " + str(rewardSum) + ", current steps: " + str(steps) + ", current score: " + str(info["score"]))
    else:
        blockedCount += 1

    done = terminated or truncated
    if done:
        maxDist = 0
        break
print("Total reward gained: " + str(rewardSum) + ", total steps: " + str(steps) + ", total score: " + str(info["score"]))
env.close()
