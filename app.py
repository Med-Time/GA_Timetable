import streamlit as st
import pandas as pd
import json
from collections import defaultdict
from data_utils import load_from_dict, load_from_json_file, total_sessions_required, build_per_day_period_map
from ga import run_ga, evaluate_fitness  # evaluate_fitness used for polishing feedback
# optional imports from ga (may raise if not present)
try:
    from ga import repair_chromosome, detect_conflicts, local_swap_improve
except Exception:
    repair_chromosome = None
    detect_conflicts = None
    local_swap_improve = None

st.set_page_config(page_title="Timetable GA", layout="wide")
st.title("Timetable Scheduling — Genetic Algorithm (Streamlit)")

# -------------------------
# Input selection controls
# -------------------------
st.sidebar.header("Input method")
input_mode = st.sidebar.radio("Choose how to provide data:", ("Use sample JSON", "Upload JSON file", "Paste JSON", "Manual entry"))

# in-memory data structures
if "classes" not in st.session_state:
    st.session_state["classes"] = []
if "rooms" not in st.session_state:
    st.session_state["rooms"] = []
if "timeslots" not in st.session_state:
    st.session_state["timeslots"] = []

# helper to clear session state when switching modes
def reset_session_inputs():
    st.session_state["classes"] = []
    st.session_state["rooms"] = []
    st.session_state["timeslots"] = []

# if user switches mode, only clear when they press a reset button to avoid accidental wipes
# -------------------------
# Load initial data depending on input mode
# -------------------------
classes = []
rooms = []
timeslots = []

if input_mode == "Use sample JSON":
    try:
        classes, rooms, timeslots = load_from_json_file("sample_data.json")
        st.info("Loaded sample_data.json")
    except Exception as e:
        st.error(f"Failed to load sample_data.json: {e}")
        st.stop()

elif input_mode == "Upload JSON file":
    uploaded = st.sidebar.file_uploader("Upload JSON", type=["json"])
    if uploaded is not None:
        try:
            data = json.load(uploaded)
            classes, rooms, timeslots = load_from_dict(data)
            st.sidebar.success("Loaded uploaded JSON")
        except Exception as e:
            st.sidebar.error(f"Failed to parse uploaded JSON: {e}")
            st.stop()
    else:
        st.sidebar.info("No file uploaded — waiting for file")

elif input_mode == "Paste JSON":
    pasted = st.sidebar.text_area("Paste JSON here (schema: classes[], rooms[], timeslots[])", height=220)
    if st.sidebar.button("Load pasted JSON"):
        if not pasted:
            st.sidebar.error("Please paste JSON then click Load pasted JSON")
        else:
            try:
                data = json.loads(pasted)
                classes, rooms, timeslots = load_from_dict(data)
                st.sidebar.success("Loaded pasted JSON")
            except Exception as e:
                st.sidebar.error(f"Failed to parse pasted JSON: {e}")

elif input_mode == "Manual entry":
    st.sidebar.markdown("Add rooms, timeslots, and subjects manually. Use **Add X** buttons to append entries.")
    # Rooms entry
    with st.sidebar.expander("Add Room (Manual)"):
        r_name = st.text_input("Room name", key="room_name")
        r_capacity = st.number_input("Capacity", min_value=1, value=75, key="room_capacity")
        r_type = st.selectbox("Room type", ("standard", "lab"), key="room_type")
        if st.button("Add room"):
            rid = len(st.session_state["rooms"])
            st.session_state["rooms"].append({"id": rid, "name": r_name or f"Room-{rid}", "capacity": int(r_capacity), "room_type": r_type})
            st.success(f"Added room {r_name or f'Room-{rid}'}")

    # Timeslot entry
    with st.sidebar.expander("Add Timeslot (Manual)"):
        # Expect label like "Mon_P1"
        ts_label = st.text_input("Timeslot label (e.g. Mon_P1)", key="ts_label")
        if st.button("Add timeslot"):
            if not ts_label:
                st.sidebar.error("Provide timeslot label")
            else:
                st.session_state["timeslots"].append(ts_label.strip())
                st.success(f"Added timeslot {ts_label.strip()}")

    # Subject entry: subject name, teacher, group (year), size, credits -> will expand to sessions
    with st.sidebar.expander("Add Subject (Manual, add credits)"):
        subj_name = st.text_input("Subject name (base)", key="subj_name")
        subj_teacher = st.text_input("Teacher name", key="subj_teacher")
        subj_group = st.text_input("Group (e.g., 2ndYear)", key="subj_group")
        subj_size = st.number_input("Class size", min_value=1, value=30, key="subj_size")
        subj_credits = st.number_input("Weekly credits (sessions) for subject", min_value=1, value=3, key="subj_credits")
        subj_room_type = st.selectbox("Room type required", ("standard", "lab"), key="subj_room_type")
        if st.button("Add subject (expand by credits)"):
            if not subj_name or not subj_teacher or not subj_group:
                st.sidebar.error("Provide subject name, teacher, and group")
            else:
                # expand into sessions SUBJ_1 ... SUBJ_k
                start_idx = len(st.session_state["classes"])
                for i in range(1, int(subj_credits) + 1):
                    sid = len(st.session_state["classes"])
                    entry = {"id": sid, "name": f"{subj_name}_{i}", "teacher": subj_teacher, "group": subj_group, "size": int(subj_size), "room_type": subj_room_type}
                    st.session_state["classes"].append(entry)
                st.success(f"Added {int(subj_credits)} sessions for subject {subj_name}")

    # Optional: buttons to clear manual inputs
    if st.sidebar.button("Reset manual entries"):
        reset_session_inputs()
        st.sidebar.success("Manual session data cleared")

    # populate from session_state
    classes = [type("C", (), c)() for c in st.session_state["classes"]]
    # convert dicts to ClassItem-like objects expected by app (simple object with attributes)
    # If data_utils.ClassItem dataclass was expected, later code uses c.__dict__ for display; having simple objects works.
    rooms = [type("R", (), r)() for r in st.session_state["rooms"]]
    timeslots = list(st.session_state["timeslots"])

# Basic validation: ensure we have at least something
if not classes or not rooms or not timeslots:
    st.warning("Incomplete input: please provide classes, rooms, and timeslots via your chosen input method.")
    st.stop()

# Sidebar: GA params
st.sidebar.header("GA Parameters")
pop_size = st.sidebar.number_input("Population size", min_value=10, max_value=1000, value=100, step=10)
generations = st.sidebar.number_input("Generations", min_value=10, max_value=5000, value=200, step=10)
cx_prob = st.sidebar.slider("Crossover prob", 0.0, 1.0, 0.5)
mut_prob = st.sidebar.slider("Mutation prob", 0.0, 1.0, 0.1)
elitism = st.sidebar.number_input("Elitism" , min_value=0, max_value=10, value=2)
seed = st.sidebar.number_input("Random seed (0 for random)", value=1)

# Input summary
st.subheader("Input Summary")
col1, col2 = st.columns(2)
col1.write(f"**# classes (sessions):** {len(classes)}")
col1.write(f"**# rooms:** {len(rooms)}")
col1.write(f"**# timeslots:** {len(timeslots)}")

# Small helper to parse timeslot label
def parse_timeslot_label(label: str):
    if "_" in label:
        parts = label.split("_", 1)
    elif "-" in label:
        parts = label.split("-", 1)
    else:
        parts = label.split(" ", 1)
    if len(parts) == 2:
        day, period = parts[0], parts[1]
    else:
        day, period = label, ""
    return day, period

# Build per-day mapping and capacity
day_order, periods_count = build_per_day_period_map(timeslots)
# per-day slot limits: total slots available per day = num_rooms * periods_in_that_day
per_day_limits_slots = {d: len(rooms) * periods_count.get(d, 0) for d in day_order}

# Capacity / anomaly check
demand = total_sessions_required(classes)
total_capacity = sum(per_day_limits_slots.values())
st.sidebar.markdown("### Capacity check")
st.sidebar.write(f"Total required sessions (demand): **{demand}**")
st.sidebar.write(f"Weekly capacity (rooms × periods): **{total_capacity}**")
if total_capacity < demand:
    short = demand - total_capacity
    st.sidebar.error(f"GLOBAL SHORTFALL: need {short} more room×timeslot slots. Suggestions: add periods or rooms.")
else:
    spare = total_capacity - demand
    st.sidebar.success(f"Capacity OK (spare slots: {spare})")

# Show raw inputs
with st.expander("Show raw input data"):
    st.write("classes:")
    # convert objects/dicts to displayable dicts
    try:
        st.dataframe(pd.DataFrame([c.__dict__ for c in classes]))
    except Exception:
        st.write(classes)
    st.write("rooms:")
    try:
        st.dataframe(pd.DataFrame([r.__dict__ for r in rooms]))
    except Exception:
        st.write(rooms)
    st.write("timeslots:")
    st.write(timeslots)
    st.write("Per-day periods (parsed):")
    st.write(periods_count)
    st.write("Per-day slot limits (rooms × periods):")
    st.write(per_day_limits_slots)

# Helper to build empty group tables
def build_empty_group_tables(classes, timeslots):
    groups = sorted({c.group for c in classes})
    day_periods = [parse_timeslot_label(ts) for ts in timeslots]
    days = []
    periods_order = []
    for d, p in day_periods:
        if d not in days:
            days.append(d)
        if p not in periods_order:
            periods_order.append(p)
    group_tables = {}
    for g in groups:
        df = pd.DataFrame([["" for _ in periods_order] for _ in days], index=days, columns=periods_order)
        group_tables[g] = df
    return group_tables, groups, days, periods_order

# Run GA button and logic
if st.button("Run GA"):
    seed_val = None if seed == 0 else int(seed)
    # subj_mode = "hard" if enforce_subject_consecutive.startswith("Hard") else "soft"
    with st.spinner("Running GA — please wait..."):
        best_chrom, best_fit = run_ga(classes, rooms, timeslots,
                                      pop_size=pop_size,
                                      generations=generations,
                                      cx_prob=cx_prob,
                                      mut_prob=mut_prob,
                                      elitism=elitism,
                                      seed=seed_val,
                                      verbose=False,
                                      per_day_limits=per_day_limits_slots)
    st.success(f"GA completed — best fitness: {best_fit}")

    # Build per-year (group) day-wise table
    group_tables, groups, days, periods_order = build_empty_group_tables(classes, timeslots)

    # Fill per-year tables using best_chrom
    for idx, gene in enumerate(best_chrom):
        timeslot_idx, room_idx = gene
        cl = classes[idx]
        ts_label = timeslots[timeslot_idx]
        day, period = parse_timeslot_label(ts_label)
        if day not in days or period not in periods_order:
            continue
        cell_text = f"{cl.name} ({rooms[room_idx].name} - {cl.teacher})"
        prev = group_tables[cl.group].at[day, period]
        if prev:
            group_tables[cl.group].at[day, period] = prev + " | CONFLICT | " + cell_text
        else:
            group_tables[cl.group].at[day, period] = cell_text

    # Display each group's day-wise timetable
    st.header("Schedules by Year (Day-wise)")
    for g in groups:
        st.subheader(f"{g} Schedule")
        st.write("Rows = Days, Columns = Periods")
        st.dataframe(group_tables[g])
        csv_label = f"{g}_timetable.csv"
        st.download_button(f"Download {g} CSV", group_tables[g].to_csv(), file_name=csv_label)

    # Room × Timeslot grid (for reference)
    st.markdown("---")
    st.subheader("Room × Timeslot Grid (for reference)")
    room_names = [r.name for r in rooms]
    timeslot_labels = timeslots
    room_grid = pd.DataFrame([["" for _ in timeslot_labels] for _ in room_names], index=room_names, columns=timeslot_labels)
    for idx, gene in enumerate(best_chrom):
        t_idx, r_idx = gene
        cl = classes[idx]
        cell_text = f"{cl.name} ({cl.teacher}) [{cl.group}]"
        prev = room_grid.iat[r_idx, t_idx]
        if prev:
            room_grid.iat[r_idx, t_idx] = prev + " | CONFLICT | " + cell_text
        else:
            room_grid.iat[r_idx, t_idx] = cell_text
    st.dataframe(room_grid)

    # Detailed conflicts summary (teacher/room/group)
    teacher_map, room_map, group_map = {}, {}, {}
    for idx, gene in enumerate(best_chrom):
        t_idx, r_idx = gene
        cl = classes[idx]
        teacher_map.setdefault((t_idx, cl.teacher), []).append({"class": cl.name, "room": rooms[r_idx].name})
        room_map.setdefault((t_idx, rooms[r_idx].name), []).append(cl.name)
        group_map.setdefault((t_idx, cl.group), []).append(cl.name)

    def collect_conflicts_map(m):
        items = []
        for k, v in m.items():
            if len(v) > 1:
                items.append({"timeslot_idx": k[0], "key": k[1], "items": v})
        return items

    conflicts = {
        "teacher": collect_conflicts_map(teacher_map),
        "room": collect_conflicts_map(room_map),
        "group": collect_conflicts_map(group_map)
    }

    st.subheader("Conflicts (detailed)")
    if any(len(v) > 0 for v in conflicts.values()):
        st.write(conflicts)
        st.warning("Conflicts detected. Use Auto-Repair to try greedy fixes, or Polish for local improvements.")
        # Auto-Repair:
        if repair_chromosome is not None:
            if st.button("Auto-Repair conflicts (greedy)"):
                with st.spinner("Attempting greedy repair..."):
                    repaired = repair_chromosome(best_chrom, classes, rooms, timeslots, per_day_limits=per_day_limits_slots)
                    repaired_fit = evaluate_fitness(repaired, classes, rooms, timeslots)
                    st.success(f"Repaired fitness: {repaired_fit}")
                    # display repaired result (rebuild group tables + room grid)
                    best_chrom = repaired
                    # rebuild group tables
                    group_tables, groups, days, periods_order = build_empty_group_tables(classes, timeslots)
                    for idx, gene in enumerate(best_chrom):
                        timeslot_idx, room_idx = gene
                        cl = classes[idx]
                        ts_label = timeslots[timeslot_idx]
                        day, period = parse_timeslot_label(ts_label)
                        if day not in days or period not in periods_order:
                            continue
                        cell_text = f"{cl.name} ({rooms[room_idx].name} - {cl.teacher})"
                        prev = group_tables[cl.group].at[day, period]
                        if prev:
                            group_tables[cl.group].at[day, period] = prev + " | CONFLICT | " + cell_text
                        else:
                            group_tables[cl.group].at[day, period] = cell_text
                    st.experimental_rerun()
        else:
            st.info("Repair function not available (not implemented in ga.py).")

        # Polish (local-swap)
        if local_swap_improve is not None:
            if st.button("Polish (local swap improvements)"):
                with st.spinner("Running local-swap polish..."):
                    polished, polished_fit = local_swap_improve(best_chrom, classes, rooms, timeslots, evaluate_fitness, iterations=500)
                    st.success(f"Polished fitness: {polished_fit}")
                    best_chrom = polished
                    st.experimental_rerun()
        else:
            st.info("Local-swap polish not available (not implemented in ga.py).")

    else:
        st.success("No hard-constraint conflicts detected — congratulations!")

    # Save combined CSV (all classes with assigned timeslot and room)
    csv_rows = []
    for idx, gene in enumerate(best_chrom):
        t_idx, r_idx = gene
        cl = classes[idx]
        csv_rows.append({
            "class_id": cl.id,
            "class_name": cl.name,
            "teacher": cl.teacher,
            "group": cl.group,
            "size": cl.size,
            "timeslot": timeslots[t_idx],
            "room": rooms[r_idx].name,
        })
    df_out = pd.DataFrame(csv_rows)
    st.download_button("Download full timetable CSV", df_out.to_csv(index=False), file_name="timetable_full.csv")
