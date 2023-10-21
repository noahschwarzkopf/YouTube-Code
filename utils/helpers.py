import numpy as np
from scipy.interpolate import make_interp_spline
import pyreadr
import pandas as pd

all_teams = ['CHC', 'BOS', 'SF', 'SEA', 'HOU', 'KC', 'BAL', 'CIN', 'MIA', 'PHI',
       'CLE', 'MIN', 'MIL', 'NYM', 'LAD', 'STL', 'DET', 'WSH', 'CWS',
       'TOR', 'ATL', 'AZ', 'COL', 'TB', 'NYY', 'OAK', 'LAA', 'PIT', 'SD',
       'TEX']

mapping = {
    'angels': 'LAA', 
    'astros': 'HOU', 
    'athletics': 'OAK', 
    'blue_jays': 'TOR', 
    'braves': 'ATL', 
    'brewers': 'MIL',
    'cardinals': 'STL', 
    'cubs': 'CHC', 
    'diamondbacks': 'AZ', 
    'dodgers': 'LAD', 
    'giants': 'SF',
    'indians': 'CLE', 
    'mariners': 'SEA', 
    'marlins': 'MIA', 
    'mets': 'NYM', 
    'nationals': 'WSH', 
    'orioles': 'BAL',
    'padres': 'SD', 
    'phillies': 'PHI', 
    'pirates': 'PIT', 
    'rangers': 'TEX', 
    'rays': 'TB', 
    'reds': 'CIN',
    'red_sox': 'BOS',
    'rockies': 'COL', 
    'royals': 'KC', 
    'tigers': 'DET', 
    'twins': 'MIN', 
    'white_sox': 'CWS', 
    'yankees': 'NYY'
}

alt = {
    'PHI': 0.00,
    'SEA':0.00,
    'SD':0.00,
    'MIA':0.00,
    'BOS':0.00,
    'WSH':0.00,
    'HOU':0.00,
    'OAK':0.00,
    'TB':0.00,
    'NYY':0.00,
    'NYM':0.00,
    'SF':0.00,
    'BAL':0.00,
    'LAA':0.00,
    'TOR':0.00,
    'LAD':0.00,
    'STL':-0.01,
    'CLE':-0.01,
    'MIL':-0.01,
    'DET':-0.01,
    'CWS':-0.01,
    'CHC':-0.01,
    'TEX':-0.01,
    'CIN':-0.02,
    'PIT':-0.02,
    'KC':-0.02,
    'MIN':-0.02,
    'ATL':-0.02,
    'AZ':-0.02,
    'COL':-0.06
    }

def spray_angle(x, y):
    """Computes the spray angle in degrees given the x and y coordinates of the spray point.

    Args:
        x (float): The x coordinate of the spray point.
        y (float): The y coordinate of the spray point.

    Returns:
        float: The spray angle in degrees.
    """
    def theta(v, w):
        return np.arccos(v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w)))

    v1 = np.array([[0, 0], [0, 10000]])
    v2 = np.array([[0, x], [0, y]])

    return np.rad2deg(theta(v1, v2))[1, 1] * np.sign(x)


def mlbam_xy_transformation(data, x="hc_x", y="hc_y", column_suffix="_", scale=2.495671):
    """Applies a transformation to the x and y columns of a DataFrame.

    Args:
        data (pandas.DataFrame): The DataFrame containing the x and y columns.
        x (str, optional): The name of the x column. Defaults to "hc_x".
        y (str, optional): The name of the y column. Defaults to "hc_y".
        column_suffix (str, optional): The suffix to append to the new column names. Defaults to "_".
        scale (float, optional): The scaling factor for the transformation. Defaults to 2.495671.

    Returns:
        pandas.DataFrame: The modified DataFrame with the new x and y columns.
    """
    data[f"{x}{column_suffix}"] = scale * (data[x] - 125)
    data[f"{y}{column_suffix}"] = scale * (199 - data[y])
    return data

# For each ball hit, use the spline to interpolate the exact distance and height needed for a homerun at the designated park

outfield_dims = pyreadr.read_r('fences.rds')[None]

outfield_dims['spray_angle'] = outfield_dims.apply(lambda x: spray_angle(x.x, x.y),1) 
outfield_dims['team_abv'] = outfield_dims['team'].apply(lambda x: mapping[x])

def home_run_needed_metrics(spray_angle, team):
    temp = outfield_dims.query(f'team_abv == "{team}"').sort_values('spray_angle')

    dist_func = make_interp_spline(temp['spray_angle'], temp['d_wall'])
    height_func = make_interp_spline(temp['spray_angle'], temp['fence_height'],k=1)

    dist = dist_func(spray_angle)
    height = height_func(spray_angle)

    return dist, height


# Use physics to determine the height at the estimated distance. First step is if estimated dist >= dist at spray angle, next is if physics height 
# is >= height at spray angle

def is_home_run_dan(hit_distance, launch_angle, hit_height, spray_angle, team):

    wall_distance, fence_height = home_run_needed_metrics(spray_angle, team)

    #Calculating V: initial velocity using estimated hit distance and launch angle
    velocity = np.sqrt(hit_distance/((np.sin(2*launch_angle*np.pi/180))/32.174))
    print(velocity)

    #Vox: Initial horizontal velocity using initial velocity and launch angle
    vox = np.cos(launch_angle/180*np.pi)*velocity
    print(vox)

    #Voy: Initial vertical velocity using initial velocity and launch angle
    voy = np.sin(launch_angle/180*np.pi)* velocity
    print(voy)
    
    #Time it takes to reach distance of mega field dimension
    time = wall_distance/vox
    print(time)

    #Calculating vertical height at distance of mega wall
    height_at_time = hit_height+(voy*time)+0.5*((-32.174)*(time**2))
    print(height_at_time)

    print(wall_distance)

    if height_at_time > fence_height:
        return 1
    else:
        return 0
    
def get_fence_height(launch_speed_fts, launch_angle_rads, plate_z, hit_distance_sc, spray_angle, team, g=-32.174):

    wall_distance, fence_height = home_run_needed_metrics(spray_angle, team)

    # calculate launch_speed_x and launch_speed_y
    launch_speed_x = launch_speed_fts * np.cos(launch_angle_rads)
    launch_speed_y = launch_speed_fts * np.sin(launch_angle_rads)
    
    # calculate total_time
    total_time = -(launch_speed_y + np.sqrt(launch_speed_y**2 + (2*g * plate_z))) / g
    
    # calculate acceleration_x
    acceleration_x = (-2*launch_speed_x / total_time) + (2*hit_distance_sc/total_time**2)
    
    # calculate time_wall
    time_wall = (-launch_speed_x + np.sqrt(launch_speed_x**2 + 2*acceleration_x*wall_distance))/acceleration_x
    
    # calculate height_at_wall
    height_at_wall = (launch_speed_y * time_wall) + (.5*g*(time_wall**2))
    
    # check if the ball clears the wall
    return fence_height

def is_home_run(launch_speed_fts, launch_angle_rads, plate_z, hit_distance_sc, spray_angle, team, g=-32.174):

    wall_distance, fence_height = home_run_needed_metrics(spray_angle, team)

    # calculate launch_speed_x and launch_speed_y
    launch_speed_x = launch_speed_fts * np.cos(launch_angle_rads)
    launch_speed_y = launch_speed_fts * np.sin(launch_angle_rads)
    
    # calculate total_time
    total_time = -(launch_speed_y + np.sqrt(launch_speed_y**2 + (2*g * plate_z))) / g
    
    # calculate acceleration_x
    acceleration_x = (-2*launch_speed_x / total_time) + (2*hit_distance_sc/total_time**2)
    
    # calculate time_wall
    time_wall = (-launch_speed_x + np.sqrt(launch_speed_x**2 + 2*acceleration_x*wall_distance))/acceleration_x
    
    # calculate height_at_wall
    height_at_wall = (launch_speed_y * time_wall) + (.5*g*(time_wall**2))
    
    # check if the ball clears the wall
    if height_at_wall > fence_height:
        return 1
    return 0

def is_home_run_2(launch_speed_fts, launch_angle_rads, plate_z, spray_angle, team, g=-32.174):

    wall_distance, fence_height = home_run_needed_metrics(spray_angle, team)

    # calculate launch_speed_x and launch_speed_y
    launch_speed_x = launch_speed_fts * np.cos(launch_angle_rads)
    launch_speed_y = launch_speed_fts * np.sin(launch_angle_rads)

    time_wall = wall_distance / launch_speed_x

    height_at_wall = plate_z + (launch_speed_y * time_wall) + (0.5 * g * (time_wall ** 2))
    
    if height_at_wall > fence_height:
        return 1
    return 0


def num_homers(launch_speed_fts, launch_angle_rads, plate_z, hit_distance_sc, spray_angle, team, g=-32.174):

    alt_team = (alt[team] * -1)

    count = 0

    for new_team in all_teams:

        wall_distance, fence_height = home_run_needed_metrics(spray_angle, new_team)

        alt_new_team = (alt[new_team] * -1)

        alt_dist_change = (alt_new_team - alt_team) + 1

        new_distance = hit_distance_sc * alt_dist_change

        # calculate launch_speed_x and launch_speed_y
        launch_speed_x = launch_speed_fts * np.cos(launch_angle_rads)
        launch_speed_y = launch_speed_fts * np.sin(launch_angle_rads)
        
        # calculate total_time
        total_time = -(launch_speed_y + np.sqrt(launch_speed_y**2 + (2*g * plate_z))) / g
        
        # calculate acceleration_x
        acceleration_x = (-2*launch_speed_x / total_time) + (2*new_distance/total_time**2)
        
        # calculate time_wall
        time_wall = (-launch_speed_x + np.sqrt(launch_speed_x**2 + 2*acceleration_x*wall_distance))/acceleration_x
        
        # calculate height_at_wall
        height_at_wall = (launch_speed_y * time_wall) + (.5*g*(time_wall**2))
        
        # check if the ball clears the wall
        if height_at_wall > fence_height:
            count += 1

    return count

def is_home_run_new_team(launch_speed_fts, launch_angle_rads, plate_z, hit_distance_sc, spray_angle, team, new_team, event, g=-32.174):

    if team == new_team:
        if event == 'home_run':
            return 1
        else:
            return 0

    else:
        alt_team = (alt[team] * -1)

        alt_new_team = (alt[new_team] * -1)

        alt_dist_change = (alt_new_team - alt_team) + 1

        hit_distance_sc = hit_distance_sc * alt_dist_change

        wall_distance, fence_height = home_run_needed_metrics(spray_angle, new_team)

        # calculate launch_speed_x and launch_speed_y
        launch_speed_x = launch_speed_fts * np.cos(launch_angle_rads)
        launch_speed_y = launch_speed_fts * np.sin(launch_angle_rads)
        
        # calculate total_time
        total_time = -(launch_speed_y + np.sqrt(launch_speed_y**2 + (2*g * plate_z))) / g
        
        # calculate acceleration_x
        acceleration_x = (-2*launch_speed_x / total_time) + (2*hit_distance_sc/total_time**2)
        
        # calculate time_wall
        time_wall = (-launch_speed_x + np.sqrt(launch_speed_x**2 + 2*acceleration_x*wall_distance))/acceleration_x
        
        # calculate height_at_wall
        height_at_wall = (launch_speed_y * time_wall) + (.5*g*(time_wall**2))
        
        # check if the ball clears the wall
        if height_at_wall > fence_height:
            return 1
        return 0
    

def is_home_run_new_team2(launch_speed_fts, launch_angle_rads, plate_z, hit_distance_sc, spray_angle, team, new_team, g=-32.174):

    if team != new_team:
        alt_team = (alt[team] * -1)

        alt_new_team = (alt[new_team] * -1)

        alt_dist_change = (alt_new_team - alt_team) + 1

        hit_distance_sc = hit_distance_sc * alt_dist_change

    wall_distance, fence_height = home_run_needed_metrics(spray_angle, new_team)

    # calculate launch_speed_x and launch_speed_y
    launch_speed_x = launch_speed_fts * np.cos(launch_angle_rads)
    launch_speed_y = launch_speed_fts * np.sin(launch_angle_rads)
    
    # calculate total_time
    total_time = -(launch_speed_y + np.sqrt(launch_speed_y**2 + (2*g * plate_z))) / g
    
    # calculate acceleration_x
    acceleration_x = (-2*launch_speed_x / total_time) + (2*hit_distance_sc/total_time**2)
    
    # calculate time_wall
    time_wall = (-launch_speed_x + np.sqrt(launch_speed_x**2 + 2*acceleration_x*wall_distance))/acceleration_x
    
    # calculate height_at_wall
    height_at_wall = (launch_speed_y * time_wall) + (.5*g*(time_wall**2))
    
    # check if the ball clears the wall
    if height_at_wall > fence_height:
        return 1
    return 0