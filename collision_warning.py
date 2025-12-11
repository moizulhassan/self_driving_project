# collision_warning.py
def time_to_collision(distance_m, relative_speed_m_s):
    # if relative_speed_m_s <= 0 (closing), compute TTC
    if relative_speed_m_s <= 0:
        return None
    # relative_speed is positive when target approaching? define:
    # We'll assume relative_speed_m_s is speed of ego - target along LOS; be careful in practice
    ttc = distance_m / relative_speed_m_s
    return ttc

def warn_if_critical(distance_m, relative_speed_m_s, ttc_threshold=3.0, dist_threshold=5.0):
    ttc = time_to_collision(distance_m, relative_speed_m_s)
    if ttc is not None and ttc < ttc_threshold:
        return True, ttc
    if distance_m < dist_threshold:
        return True, ttc
    return False, ttc

