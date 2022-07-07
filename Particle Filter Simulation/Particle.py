import math
import random

import numpy as np


class Particle:
    def __init__(self, x, y, heading, w=1):
        self.x = x
        self.y = y
        self.h = heading
        self.w = w

    def move_by(self, move_est, heading):
        if self.h != 0:
            print(self.h)
        self.x += move_est * np.cos(math.radians(heading))
        self.y += move_est * np.sin(math.radians(heading))


def bisect_left(a, x, lo=0, hi=None, *, key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(i, x) will
    insert just before the leftmost x already there.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if key(a[mid]) < x:
                lo = mid + 1
            else:
                hi = mid
    return lo


class WeightedDistribution:
    def __init__(self, state):
        accum = 0.0
        self.state = [p for p in state if p.w > 0]
        self.distribution = []
        for x in self.state:
            accum += x.w
            self.distribution.append(accum)

    def pick(self):
        try:
            return self.state[bisect_left(self.distribution, random.uniform(0, 1))]
        except IndexError:
            # Happens when all particles are improbable w=0
            return


def initialize_particles(estx, esty, esth, particle_count):
    particles = []
    particle_num = 1

    while particle_num < particle_count:
        r = random.randint(0, 15)
        angle = np.random.uniform(-math.pi/360, math.pi/360)
        particle_heading = angle + esth
        particle_x = (r * math.cos(angle)) + estx
        particle_y = (r * math.sin(angle)) + esty
        particles.append(Particle(particle_x, particle_y, heading=esth))
        particle_num = particle_num + 1

    return particles


def get_nearest_tree(trees, x, y):
    min_dist = 99999
    min_tree = []
    for tree in trees:
        dist = np.sqrt((tree.tree_x - x) ** 2 + (tree.tree_y - y) ** 2)
        if dist < min_dist:
            min_dist = dist
            min_tree = tree

    tree_x_from_particle = min_tree.tree_x - x
    tree_y_from_particle = min_tree.tree_y - y
    # angle = math.atan2(tree_y_from_particle, tree_x_from_particle)

    return min_dist, tree_x_from_particle, tree_y_from_particle, min_tree


def w_gauss(r_d, p_d):
    error = r_d - p_d
    g_1 = np.exp(-np.power(error, 2))
    g = g_1
    return g


def compute_mean_point(particles):
    m_x, m_y, m_count = 0, 0, 0
    for p in particles:
        m_count += p.w
        m_x += p.x * p.w
        m_y += p.y * p.w

    if m_count == 0:
        return -1, -1, False

    m_x /= m_count
    m_y /= m_count

    m_count = 0
    for p in particles:
        if np.sqrt((p.x - m_x) ** 2 + (p.y - m_y) ** 2) < 1:
            m_count += 1

    return m_x, m_y, m_count


def particle_filter(detected_trees, particles, tree_distances, orchard):
    print("particle")

    for p in particles:
        if p.y > orchard.orchard_starty + 200 or p.y < orchard.orchard_starty + 20 or p.x < orchard.orchard_startx:
            p.w = 0
        else:
            # importance weight
            p.w = 1 / len(particles)
            for ti, t in enumerate(detected_trees):
                robot_tree_dist = tree_distances[ti]
                particle_tree_dist = np.sqrt(np.power((p.x - t[0]), 2) + np.power((p.y - t[1]), 2))
                p.w = p.w * w_gauss(robot_tree_dist, particle_tree_dist)
                if p.w == math.inf:
                    print("inf")

    m_x, m_y, m_confident = compute_mean_point(particles)
    new_particles = []

    # Normalise weights
    nu = sum(p.w for p in particles)
    if nu:
        for p in particles:
            p.w = p.w / nu

    # create a weighted distribution, for fast picking
    dist = WeightedDistribution(particles)
    i = 0
    for _ in particles:
        i = i + 1
        p = dist.pick()
        new_particle = Particle(p.x, p.y, heading=0, w=1)

        # 1 percent randomization
        pick_filter = random.uniform(0, 1)
        if pick_filter > 0.99 and p.w < 0.25:
            r = random.randint(0, 20)
            angle = random.uniform(0, 2 * math.pi)
            particle_x = (r * math.cos(angle)) + m_x
            particle_y = (r * math.sin(angle)) + m_y
            new_particle = Particle(particle_x, particle_y, heading=0)

        new_particles.append(new_particle)
    m_x, m_y, m_confident = compute_mean_point(new_particles)
    return new_particles, m_x, m_y


def get_trees_from_robot(robot, tree_matrix):
    # translation
    translation_matrix = np.matrix([[1, 0, -robot.robot_x], [0, 1, -robot.robot_y], [0, 0, 1]])
    trees_translation = np.matmul(translation_matrix, tree_matrix)
    # rotation
    rotation_matrix = np.matrix([[np.cos(np.radians(robot.robot_angle)),
                                  -np.sin(np.radians(robot.robot_angle)), 0],
                                 [np.sin(np.radians(robot.robot_angle)),
                                  np.cos(np.radians(robot.robot_angle)), 0],
                                 [0, 0, 1]])

    trees_rotated = np.matmul(rotation_matrix, trees_translation)

    return trees_rotated


def particle_filter_sensor_fusion(detected_trees, particles, tree_distances, orchard):
    # print("particle")

    for p in particles:
        if p.y > orchard.orchard_starty + 200 or p.y < orchard.orchard_starty + 20 or p.x < orchard.orchard_startx:
            p.w = 0
        else:
            # importance weight
            p.w = 1 / len(particles)
            for ti, t in enumerate(detected_trees):
                robot_tree_dist = tree_distances[ti]
                particle_tree_dist = np.sqrt(np.power((p.x - t[0]), 2) + np.power((p.y - t[1]), 2))
                p.w = p.w * w_gauss(robot_tree_dist, particle_tree_dist)
                if p.w == math.inf:
                    print("inf")
                    input()

    m_x, m_y, m_confident = compute_mean_point(particles)
    new_particles = []

    # Normalise weights
    nu = sum(p.w for p in particles)
    if nu:
        for p in particles:
            p.w = p.w / nu
    else:
        print(nu)
        p_n = 0
        for p in particles:
            print(p.w)
        print(p_n)
        input()

    # create a weighted distribution, for fast picking
    dist = WeightedDistribution(particles)
    i = 0
    for _ in particles:
        i = i + 1
        p = dist.pick()
        new_particle = Particle(p.x, p.y, heading=0, w=1)
        # 1 percent randomization
        pick_filter = random.uniform(0, 1)
        if pick_filter > 0.99 and p.w < 0.25:
            r = random.randint(0, 20)
            angle = random.uniform(0, 2 * math.pi)
            particle_x = (r * math.cos(angle)) + m_x
            particle_y = (r * math.sin(angle)) + m_y
            new_particle = Particle(particle_x, particle_y, heading=0)
        new_particles.append(new_particle)
    m_x, m_y, m_confident = compute_mean_point(new_particles)
    return new_particles, m_x, m_y
