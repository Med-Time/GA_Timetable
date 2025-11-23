import random
from typing import List, Tuple, Dict
from data_utils import ClassItem, Room
from collections import defaultdict
import math
import random
# Gene: (timeslot_index, room_index)

def random_gene(timeslots, rooms):
    return (random.randrange(len(timeslots)), random.randrange(len(rooms)))

def create_random_chromosome(classes, rooms, timeslots):
    return [random_gene(timeslots, rooms) for _ in classes]

# ---------- Subject-per-day consecutive constraint helpers ----------

def subject_base(name: str):
    """
    Extract base subject name from class name like 'COA_1' -> 'COA'.
    If names are already base subjects, it returns as-is.
    """
    if "_" in name:
        return name.split("_", 1)[0]
    return name

def get_subject_weekly_counts(classes):
    """
    Count how many sessions each base-subject has in the whole classes list (i.e., weekly sessions).
    Returns dict: subject -> count (int).
    This is used to determine which subjects have >3 sessions/week.
    """
    counts = {}
    for cl in classes:
        subj = subject_base(cl.name)
        counts[subj] = counts.get(subj, 0) + 1
    return counts

def _build_day_order_and_tsindex_map(timeslots):
    """
    Build:
     - day_periods: dict day -> list of (period_label, ts_idx) in appearance order
     - tsidx_to_order: dict ts_idx -> (day, order_index_within_day)
    """
    day_periods = {}
    for ts_idx, label in enumerate(timeslots):
        if "_" in label:
            day, period = label.split("_", 1)
        elif "-" in label:
            day, period = label.split("-", 1)
        else:
            parts = label.split(" ", 1)
            day = parts[0]
            period = parts[1] if len(parts) > 1 else ""
        day_periods.setdefault(day, []).append((period, ts_idx))
    tsidx_to_order = {}
    for d in day_periods:
        # already in appearance order by ts_idx, but ensure sort by ts_idx
        day_periods[d].sort(key=lambda x: x[1])
        for order, (_p, tsidx) in enumerate(day_periods[d]):
            tsidx_to_order[tsidx] = (d, order)
    return day_periods, tsidx_to_order

def find_subject_day_violations(chromosome, classes, timeslots):
    """
    Return list of violations: (subject, day, sorted_list_of_period_orders_on_that_day)
    ONLY subjects with weekly_count > 3 are checked; subjects with <=3 sessions/week are ignored.
    """
    # compute weekly counts
    weekly_counts = get_subject_weekly_counts(classes)

    # build ts index -> (day, order)
    day_periods, tsidx_to_order = _build_day_order_and_tsindex_map(timeslots)

    subj_day_map = {}
    for idx, gene in enumerate(chromosome):
        ts_idx, _rm = gene
        if ts_idx not in tsidx_to_order:
            continue
        day, order = tsidx_to_order[ts_idx]
        subj = subject_base(classes[idx].name)
        subj_day_map.setdefault((subj, day), []).append(order)

    violations = []
    for (subj, day), orders in subj_day_map.items():
        weekly = weekly_counts.get(subj, 0)
        # Only enforce the consecutive-block rule for subjects with weekly sessions > 3
        if weekly <= 3:
            continue
        if len(orders) <= 1:
            continue  # single session on a day is always ok
        orders_sorted = sorted(orders)
        # check if these orders form a single consecutive block
        is_consecutive = all(orders_sorted[i] + 1 == orders_sorted[i+1] for i in range(len(orders_sorted)-1))
        if not is_consecutive:
            violations.append((subj, day, orders_sorted))
    return violations

SUBJECT_DAY_VIOL_PEN = 2000  # large penalty — treat as hard constraint


def parse_day_of_timeslot_list(timeslots):
    # return mapping timeslot_idx -> day (expects 'Mon_P1' style)
    day_of_ts = {}
    for i, label in enumerate(timeslots):
        if "_" in label:
            day = label.split("_",1)[0]
        elif "-" in label:
            day = label.split("-",1)[0]
        else:
            day = label.split(" ",1)[0]
        day_of_ts[i] = day
    return day_of_ts

def detect_conflicts(chromosome, classes, rooms, timeslots):
    teacher_map, room_map, group_map = {}, {}, {}
    for idx, gene in enumerate(chromosome):
        t_idx, r_idx = gene
        cl = classes[idx]
        teacher_map.setdefault((t_idx, cl.teacher), []).append((idx, cl, rooms[r_idx].name))
        room_map.setdefault((t_idx, r_idx), []).append((idx, cl))
        group_map.setdefault((t_idx, cl.group), []).append((idx, cl))
    conflicts = {
        "teacher": [(k, v) for k,v in teacher_map.items() if len(v)>1],
        "room": [(k, v) for k,v in room_map.items() if len(v)>1],
        "group": [(k, v) for k,v in group_map.items() if len(v)>1]
    }
    return conflicts

def build_occupancy(chromosome):
    return {(g[0], g[1]): True for g in chromosome}

def would_create_subject_day_violation(chromosome, classes, timeslots, class_idx, candidate_ts):
    """
    Simulate moving class_idx to candidate_ts. Return True if that move would create a
    subject-day non-consecutive violation **for that subject**, given the weekly counts rule (>3).
    """
    # quick check: if subject has weekly_count <= 3 then it's never restricted
    subj = subject_base(classes[class_idx].name)
    weekly_counts = get_subject_weekly_counts(classes)
    if weekly_counts.get(subj, 0) <= 3:
        return False

    # simulate placing class_idx at candidate_ts (room unchanged for simulation)
    sim = list(chromosome)
    _, curr_room = sim[class_idx]
    sim[class_idx] = (candidate_ts, curr_room)

    # compute violations on simulated chromosome
    violations = find_subject_day_violations(sim, classes, timeslots)
    # if violations include this subject on the candidate day, then move is bad
    # derive candidate_day
    if "_" in timeslots[candidate_ts]:
        candidate_day = timeslots[candidate_ts].split("_", 1)[0]
    elif "-" in timeslots[candidate_ts]:
        candidate_day = timeslots[candidate_ts].split("-", 1)[0]
    else:
        candidate_day = timeslots[candidate_ts].split(" ", 1)[0]
    for v in violations:
        if v[0] == subj and v[1] == candidate_day:
            return True
    return False

def find_free_slot(chromosome, classes, rooms, timeslots, class_idx,
                   per_day_limits=None, timeslot_day_map=None, rooms_info=None):
    """
    Greedy: return (ts,room) free slot that respects per_day_limits for the class's day (if provided).
    per_day_limits: dict day->max_periods_allowed (global per-day Pn), used with current occupancy counts
    timeslot_day_map: mapping idx->day
    rooms_info: list of Room objects for capacity/type checks (optional)
    """
    occupied = build_occupancy(chromosome)
    # compute current counts per day for the class's group? We treat per-day global cap (all classes)
    day_counts = defaultdict(int)
    for (t, r) in occupied.keys():
        day = timeslot_day_map[t]
        day_counts[day] += 1

    # try to find a slot that doesn't violate per_day_limits
    for ts in range(len(timeslots)):
        day = timeslot_day_map[ts]
        # if per-day limit exists, check whether adding would exceed
        if per_day_limits and day_counts.get(day, 0) >= per_day_limits.get(day, float('inf')):
            continue
        for rm in range(len(rooms)):
            if (ts, rm) not in occupied:
                # optional: capacity/type checks
                if rooms_info:
                    cl = classes[class_idx]
                    room = rooms_info[rm]
                    if getattr(room, 'capacity', 1) < getattr(cl, 'size', 0):
                        # skip if room too small
                        continue
                    # if class has room_type and room doesn't match, skip
                    if getattr(cl, 'room_type', None) and getattr(room, 'room_type', None):
                        if cl.room_type != room.room_type:
                            continue
                if would_create_subject_day_violation(chromosome, classes, timeslots, class_idx, ts):
                    continue  # skip this candidate slot
                return (ts, rm)
    return None

def repair_chromosome(chromosome, classes, rooms, timeslots,
                      per_day_limits=None, max_attempts=500):
    """
    Attempt to remove teacher/room/group clashes by relocating conflicting classes greedily.
    Returns repaired chromosome (may be same if no repair possible).
    """
    timeslot_day_map = parse_day_of_timeslot_list(timeslots)
    chrom = list(chromosome)
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        conflicts = detect_conflicts(chrom, classes, rooms, timeslots)
        total_conflicts = sum(len(v) for v in conflicts['teacher']) + sum(len(v) for v in conflicts['room']) + sum(len(v) for v in conflicts['group'])
        if total_conflicts == 0:
            break
        moved = False
        # process teacher conflicts first
        for (t_idx, teacher), entries in conflicts['teacher']:
            # keep first, move rest
            for entry in entries[1:]:
                idx_to_move = entry[0]
                free = find_free_slot(chrom, classes, rooms, timeslots, idx_to_move,
                                      per_day_limits=per_day_limits, timeslot_day_map=timeslot_day_map, rooms_info=rooms)
                if free:
                    chrom[idx_to_move] = free
                    moved = True
        # then room conflicts
        for (t_idx, room_idx), entries in conflicts['room']:
            for entry in entries[1:]:
                idx_to_move = entry[0]
                free = find_free_slot(chrom, classes, rooms, timeslots, idx_to_move,
                                      per_day_limits=per_day_limits, timeslot_day_map=timeslot_day_map, rooms_info=rooms)
                if free:
                    chrom[idx_to_move] = free
                    moved = True
        # then group conflicts
        for (t_idx, group), entries in conflicts['group']:
            for entry in entries[1:]:
                idx_to_move = entry[0]
                free = find_free_slot(chrom, classes, rooms, timeslots, idx_to_move,
                                      per_day_limits=per_day_limits, timeslot_day_map=timeslot_day_map, rooms_info=rooms)
                if free:
                    chrom[idx_to_move] = free
                    moved = True
        if not moved:
            break
    return chrom

def local_swap_improve(chromosome, classes, rooms, timeslots, fitness_fn, iterations=500):
    best = list(chromosome)
    best_score = fitness_fn(best, classes, rooms, timeslots)
    n = len(chromosome)
    for _ in range(iterations):
        i, j = random.sample(range(n), 2)
        c = list(best)
        c[i], c[j] = c[j], c[i]
        s = fitness_fn(c, classes, rooms, timeslots)
        if s > best_score:
            best, best_score = c, s
    return best, best_score


# ---------- Fitness function ----------
def evaluate_fitness(chromosome, classes: List[ClassItem], rooms: List[Room], timeslots: List[str]) -> int:
    PEN_HARD = 1000
    PEN_ROOM_CAP = 10
    penalty = 0

    teacher_map: Dict[Tuple[int,str], List[int]] = {}
    room_map: Dict[Tuple[int,int], List[int]] = {}
    group_map: Dict[Tuple[int,str], List[int]] = {}

    for idx, gene in enumerate(chromosome):
        timeslot_idx, room_idx = gene
        cl = classes[idx]
        teacher_key = (timeslot_idx, cl.teacher)
        room_key = (timeslot_idx, room_idx)
        group_key = (timeslot_idx, cl.group)

        teacher_map.setdefault(teacher_key, []).append(idx)
        room_map.setdefault(room_key, []).append(idx)
        group_map.setdefault(group_key, []).append(idx)

        room = rooms[room_idx]
        if room.capacity < cl.size:
            penalty += PEN_ROOM_CAP * (cl.size - room.capacity)

    for m in (teacher_map, room_map, group_map):
        for key, lst in m.items():
            if len(lst) > 1:
                violations = len(lst) - 1
                penalty += PEN_HARD * violations
    # subject-per-day consecutive violations
    try:
        subj_day_violations = find_subject_day_violations(chromosome, classes, timeslots)
        penalty += SUBJECT_DAY_VIOL_PEN * len(subj_day_violations)
    except Exception:
        # find_subject_day_violations or SUBJECT_DAY_VIOL_PEN not defined — skip this penalty
        pass

    return -penalty


# ---------- GA operators ----------
def tournament_selection(population, fitnesses, k=3):
    best = None
    best_f = None
    n = len(population)
    for _ in range(k):
        i = random.randrange(n)
        f = fitnesses[i]
        if best is None or f > best_f:
            best = population[i]
            best_f = f
    return list(best)

def uniform_crossover(p1, p2, cx_prob=0.5):
    n = len(p1)
    c1 = []
    for i in range(n):
        if random.random() < cx_prob:
            c1.append(p1[i])
        else:
            c1.append(p2[i])
    return c1

def mutate(chromosome, timeslots, rooms, pm=0.1):
    n = len(chromosome)
    for i in range(n):
        if random.random() < pm:
            timeslot_idx, room_idx = chromosome[i]
            if random.random() < 0.5:
                timeslot_idx = random.randrange(len(timeslots))
            else:
                room_idx = random.randrange(len(rooms))
            chromosome[i] = (timeslot_idx, room_idx)


# ---------- GA main ----------
def run_ga(classes, rooms, timeslots,
           pop_size=100, generations=200,
           cx_prob=0.5, mut_prob=0.1, elitism=2, seed=None, verbose=False,
           per_day_limits=None):
    """
    Genetic Algorithm main loop.

    per_day_limits: optional dict mapping day (e.g. "Mon") -> max_slots_for_day (int).
                    Used by repair_chromosome/find_free_slot (if implemented) to avoid
                    moving classes into days that already reached capacity.
    """
    if seed is not None:
        random.seed(seed)

    # initialize population
    population = [create_random_chromosome(classes, rooms, timeslots) for _ in range(pop_size)]
    fitnesses = [evaluate_fitness(ind, classes, rooms, timeslots) for ind in population]


    best_overall = None
    best_fitness = float("-inf")

    for gen in range(generations):
        paired = list(zip(population, fitnesses))
        paired.sort(key=lambda x: x[1], reverse=True)
        elites = [list(x[0]) for x in paired[:elitism]]

        # update best
        if paired and paired[0][1] > best_fitness:
            best_fitness = paired[0][1]
            best_overall = list(paired[0][0])

        # produce next generation
        newpop = elites.copy()
        while len(newpop) < pop_size:
            p1 = tournament_selection(population, fitnesses, k=3)
            p2 = tournament_selection(population, fitnesses, k=3)
            child = uniform_crossover(p1, p2, cx_prob=cx_prob)
            mutate(child, timeslots, rooms, pm=mut_prob)

            # quick greedy repair of child (if repair functions available)
            try:
                child = repair_chromosome(child, classes, rooms, timeslots, per_day_limits=per_day_limits)
            except NameError:
                # repair_chromosome not defined — skip repair
                pass

            newpop.append(child)

        population = newpop
        fitnesses = [evaluate_fitness(ind, classes, rooms, timeslots) for ind in population]


        if verbose and (gen % max(1, generations // 10) == 0 or gen == generations - 1):
            print(f"Gen {gen+1}/{generations} best fitness: {best_fitness}")

        # early stop if perfect solution (zero penalty)
        if best_fitness == 0:
            if verbose:
                print(f"Perfect solution found at generation {gen+1}")
            break

    # finalize: try repair + polish on the best found
    if best_overall is not None:
        try:
            best_repaired = repair_chromosome(list(best_overall), classes, rooms, timeslots, per_day_limits=per_day_limits)
            best_repaired_fit = evaluate_fitness(best_repaired, classes, rooms, timeslots)

            if best_repaired_fit > best_fitness:
                best_overall, best_fitness = best_repaired, best_repaired_fit
        except NameError:
            pass

        try:
            best_polished, best_polished_fit = local_swap_improve(best_overall, classes, rooms, timeslots, evaluate_fitness, iterations=500)
            if best_polished_fit > best_fitness:
                best_overall, best_fitness = best_polished, best_polished_fit
        except NameError:
            pass

    return best_overall, best_fitness
