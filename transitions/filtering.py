import numpy as np
from scipy import stats

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2' """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    rads = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return math.degrees(rads)


def translate_vector(vec):
    start = vec[0]
    end = vec[1]
    new_end = end - start
    return new_end


def get_point(strpoint):
    point = strpoint.replace('(', '').replace(')', '').split(',')
    point = [int(p) for p in point]
    # return np.array(point)
    return point


def estimate_handover(source, dest, cur_path, len_cur_path, angle_lim=30, cutoff_long=False, cutoff_short=False):
    remaining_durations = []
    remaining_transitions = []
    remaining_labels = []
    cur_dist = 0
    for i, point in enumerate(cur_path[:-1]):
        cur_dist += math.hypot(point[0] - cur_path[i + 1][0], point[1] - cur_path[i + 1][1])

    cur_vector = translate_vector((cur_path[0], cur_path[-1]))

    for path, label, transition in traces[source][:-1]:
        # if label == dest:
        # * Calculate current target trace distance
        # ? Can we calculate it only the first time we encounter the trace?
        dist = 0
        for i, point in enumerate(path[:-1]):
            dist += math.hypot(point[0] - path[i + 1][0], point[1] - path[i + 1][1])

        # * Calculate approximate remaining distance
        end = path[-1]
        cur_point = cur_path[-1]
        min_remaining = math.hypot(end[0] - cur_point[0], end[1] - cur_point[1])

        # * Remove paths shorter than ours already is in either time or distance
        if len(path) < len_cur_path or dist < cur_dist:
            continue
        # * Remove paths that have a distance shorter than our minimum possible distance
        if cutoff_short and 0.95 * (cur_dist + min_remaining) > dist:
            continue
        # # * If min_remaining relatively small AND len difference between cur_path and path too large, discard point?
        # # * Logic being yes, we are close to the exit but the other path took the trace very slowly
        # # ? Can this be improved?
        # # ? Specifically, in calculation of distance to path relations
        if cutoff_long and min_remaining / cur_dist < 0.20:
            if len_cur_path / len(path) < 1 - min_remaining / cur_dist:
                continue
        # * Calculate angle between current and existing path
        # * If angle is too large (above given threshold) then the existing trace has a different heading
        path_vector = translate_vector((path[0], path[-1]))
        angle = angle_between(cur_vector, path_vector)
        if angle > angle_lim:  # ? Angle between vectors is always <180 or angle < -angle_lim:
            continue
        remaining_durations.append(len(path))
        remaining_transitions.append(transition)
        remaining_labels.append(label)
    # ! DEAL WITH THE CASE WHERE EVERYTHING IS FILTERED OUT
    if len(remaining_durations) != 0:
        # TODO Calculate mean transition distance
        return stats.mode(remaining_durations)[0][0], stats.mode(remaining_transitions)[0][0], stats.mode(remaining_labels)[0][0]
        # return int(statistics.median(remaining_durations)), int(statistics.median(remaining_transitions))
    return -1, 0, -1