from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


DIRECTIONS: dict[str, tuple[int, int]] = {
    "up": (0, 1),
    "right": (1, 0),
    "down": (0, -1),
    "left": (-1, 0),
}

CAUSE_LABELS = {
    "snake-collision": "Koerperkollision",
    "snake-self-collision": "Selbstkollision",
    "head-collision": "Head-to-head",
    "wall-collision": "Wand",
    "out-of-health": "Verhungert",
}

TAG_LABELS = {
    "adjacent_enemy_head": "nah am gegnerischen Kopf",
    "corridor": "Korridor",
    "dead_end": "Sackgasse",
    "dead_end_alternative_available": "Sackgassen-Alternative verpasst",
    "edge": "am Rand",
    "fatal_hazard": "Hazard waere fatal",
    "fatal_opponent_body": "sichtbarer Gegnerkoerper",
    "fatal_out_of_health": "kein Health-Puffer",
    "fatal_self_body": "eigener Koerper",
    "fatal_wall": "ausserhalb des Boards",
    "food": "Food-Ziel",
    "hazard": "Hazard",
    "head_attack_opportunity": "Kopf-Angriffschance",
    "head_risk": "Head-to-head-Risiko",
    "head_safe_alternative_available": "kopfsichere Alternative verfuegbar",
    "high_latency": "hohe Latenz",
    "low_health_no_food_progress": "bei wenig Health kein Food-Fortschritt",
    "low_space": "wenig erreichbarer Raum",
    "missed_space": "deutlich mehr Raum verfuegbar",
    "no_food_path_critical": "kritisch wenig Health ohne Food-Pfad",
    "no_food_progress_critical": "kritisch wenig Health ohne Food-Fortschritt",
    "opponent_tail_on_food": "Gegnertail auf Food",
    "safer_alternative_available": "sicherere Alternative verfuegbar",
    "space_less_than_length": "Raum kleiner als Laenge",
    "tiny_region": "sehr kleiner Bereich",
    "timeout_risk": "Timeout-Risiko",
}


@dataclass(frozen=True)
class LogChoice:
    path: Path
    score: tuple[int, int, int, int, str]


@dataclass
class MoveEval:
    direction: str
    target: tuple[int, int] | None
    fatal_reasons: list[str]
    risk_tags: list[str]
    reachable_space: int
    open_neighbors: int
    wall_distance: int
    food_distance: int | None
    head_risk: list[str]
    head_attack: list[str]
    nearest_enemy_head_distance: int | None
    hits_food: bool


def point_tuple(point: Any) -> tuple[int, int] | None:
    if not isinstance(point, dict):
        return None
    x = point.get("x")
    y = point.get("y")
    if isinstance(x, int) and isinstance(y, int):
        return x, y
    return None


def in_bounds(point: tuple[int, int] | None, width: int, height: int) -> bool:
    if point is None:
        return False
    x, y = point
    return 0 <= x < width and 0 <= y < height


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def move_target(head: tuple[int, int] | None, direction: str) -> tuple[int, int] | None:
    if head is None or direction not in DIRECTIONS:
        return None
    dx, dy = DIRECTIONS[direction]
    return head[0] + dx, head[1] + dy


def all_snakes(state: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(state, dict):
        return []
    seen: set[str] = set()
    snakes: list[dict[str, Any]] = []
    board = state.get("board") or {}
    for snake in board.get("snakes") or []:
        snake_id = snake.get("id")
        if snake_id is None or snake_id in seen:
            continue
        seen.add(snake_id)
        snakes.append(snake)
    you = state.get("you")
    if isinstance(you, dict):
        you_id = you.get("id")
        if you_id is not None and you_id not in seen:
            snakes.append(you)
    return snakes


def snake_name(snake: dict[str, Any] | None) -> str:
    if not isinstance(snake, dict):
        return "unknown"
    return str(snake.get("name") or snake.get("id") or "unknown")


def snake_alive(snake: dict[str, Any], width: int, height: int) -> bool:
    return (
        snake.get("elimination_event") is None
        and in_bounds(point_tuple(snake.get("head")), width, height)
    )


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARN: {path}: konnte JSON nicht lesen ({exc})")
        return None
    if not isinstance(data, dict):
        print(f"WARN: {path}: Top-Level JSON ist kein Objekt")
        return None
    return data


def choose_log_paths(paths: list[Path], include_duplicates: bool) -> tuple[list[Path], int]:
    if include_duplicates:
        return paths, 0

    best_by_game: dict[str, LogChoice] = {}
    skipped = 0
    for path in paths:
        data = load_json(path)
        if data is None:
            skipped += 1
            continue
        game_id = str(data.get("game_id") or path.stem)
        moves = data.get("moves") or []
        score = (
            1 if data.get("result") else 0,
            1 if moves else 0,
            len(moves),
            1 if data.get("end_state") else 0,
            path.name,
        )
        current = best_by_game.get(game_id)
        if current is None or score > current.score:
            best_by_game[game_id] = LogChoice(path=path, score=score)
    selected = sorted((choice.path for choice in best_by_game.values()), key=lambda p: p.name)
    return selected, max(0, len(paths) - len(selected) - skipped)


def infer_you_id(data: dict[str, Any]) -> str | None:
    result = data.get("result") or {}
    if result.get("you"):
        return str(result["you"])
    for state_key in ("end_state", "start_state"):
        state = data.get(state_key)
        if isinstance(state, dict) and isinstance(state.get("you"), dict):
            return state["you"].get("id")
    for move in data.get("moves") or []:
        state = move.get("state")
        if isinstance(state, dict) and isinstance(state.get("you"), dict):
            return state["you"].get("id")
    return None


def result_snake(data: dict[str, Any], you_id: str | None) -> dict[str, Any] | None:
    if you_id is None:
        return None
    result = data.get("result") or {}
    for snake in result.get("snakes") or []:
        if snake.get("id") == you_id:
            return snake
    end_state = data.get("end_state")
    if isinstance(end_state, dict):
        you = end_state.get("you")
        if isinstance(you, dict) and you.get("id") == you_id:
            return you
        for snake in all_snakes(end_state):
            if snake.get("id") == you_id:
                return snake
    return None


def infer_outcome(data: dict[str, Any], you_id: str | None) -> str:
    result = data.get("result") or {}
    if result.get("outcome"):
        return str(result["outcome"])
    you = result_snake(data, you_id)
    if you and you.get("elimination_event"):
        return "loss"
    end_state = data.get("end_state")
    if isinstance(end_state, dict) and you_id:
        width = int((end_state.get("board") or {}).get("width") or 0)
        height = int((end_state.get("board") or {}).get("height") or 0)
        alive = [
            snake.get("id")
            for snake in all_snakes(end_state)
            if snake_alive(snake, width, height)
        ]
        if alive == [you_id]:
            return "win"
        if you_id in alive:
            return "draw"
    return "unknown"


def infer_game_turn(data: dict[str, Any]) -> int | None:
    result = data.get("result") or {}
    if isinstance(result.get("turn"), int):
        return result["turn"]
    end_state = data.get("end_state")
    if isinstance(end_state, dict) and isinstance(end_state.get("turn"), int):
        return end_state["turn"]
    moves = data.get("moves") or []
    if moves and isinstance(moves[-1].get("turn"), int):
        return moves[-1]["turn"]
    return None


def collect_opponent_profiles(
    data: dict[str, Any], you_id: str | None
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    profiles: dict[str, dict[str, Any]] = {}
    opponent_names: set[str] = set()

    def add_snake(snake: dict[str, Any], turn: int | None = None) -> None:
        snake_id = snake.get("id")
        if not snake_id or snake_id == you_id:
            return
        name = snake_name(snake)
        opponent_names.add(name)
        profile = profiles.setdefault(
            snake_id,
            {
                "id": snake_id,
                "name": name,
                "first_turn": turn,
                "last_turn": turn,
                "max_length": 0,
                "max_health": None,
                "seen_turns": 0,
            },
        )
        profile["name"] = name
        profile["max_length"] = max(profile["max_length"], int(snake.get("length") or 0))
        health = snake.get("health")
        if isinstance(health, int):
            if profile["max_health"] is None:
                profile["max_health"] = health
            else:
                profile["max_health"] = max(profile["max_health"], health)
        if turn is not None:
            profile["seen_turns"] += 1
            if profile["first_turn"] is None:
                profile["first_turn"] = turn
            else:
                profile["first_turn"] = min(profile["first_turn"], turn)
            if profile["last_turn"] is None:
                profile["last_turn"] = turn
            else:
                profile["last_turn"] = max(profile["last_turn"], turn)

    for state_key in ("start_state", "end_state"):
        state = data.get(state_key)
        if isinstance(state, dict):
            add_turn = state.get("turn") if isinstance(state.get("turn"), int) else None
            for snake in all_snakes(state):
                add_snake(snake, add_turn)

    for move in data.get("moves") or []:
        state = move.get("state")
        if isinstance(state, dict):
            turn = state.get("turn") if isinstance(state.get("turn"), int) else move.get("turn")
            for snake in all_snakes(state):
                add_snake(snake, turn if isinstance(turn, int) else None)

    result = data.get("result") or {}
    for snake in result.get("snakes") or []:
        add_snake(snake, result.get("turn") if isinstance(result.get("turn"), int) else None)

    return profiles, opponent_names


def build_context(state: dict[str, Any]) -> dict[str, Any]:
    board = state.get("board") or {}
    width = int(board.get("width") or 0)
    height = int(board.get("height") or 0)
    you = state.get("you") or {}
    you_id = you.get("id")

    active_snakes = [
        snake
        for snake in all_snakes(state)
        if snake.get("id") == you_id or snake_alive(snake, width, height)
    ]

    body_cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    solid_body_cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    tail_cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for snake in active_snakes:
        snake_id = snake.get("id")
        body = snake.get("body") or []
        for index, raw_point in enumerate(body):
            point = point_tuple(raw_point)
            if not in_bounds(point, width, height):
                continue
            is_tail = index == len(body) - 1
            hit = {
                "snake_id": snake_id,
                "snake_name": snake_name(snake),
                "index": index,
                "is_tail": is_tail,
            }
            body_cells[point].append(hit)
            if is_tail:
                tail_cells[point].append(hit)
            else:
                solid_body_cells[point].append(hit)

    obstacles = set(solid_body_cells)
    food = {
        point
        for point in (point_tuple(raw_food) for raw_food in board.get("food") or [])
        if in_bounds(point, width, height)
    }
    hazards = {
        point
        for point in (point_tuple(raw_hazard) for raw_hazard in board.get("hazards") or [])
        if in_bounds(point, width, height)
    }
    ruleset = ((state.get("game") or {}).get("ruleset") or {}).get("settings") or {}

    return {
        "width": width,
        "height": height,
        "you": you,
        "you_id": you_id,
        "active_snakes": active_snakes,
        "body_cells": body_cells,
        "solid_body_cells": solid_body_cells,
        "tail_cells": tail_cells,
        "obstacles": obstacles,
        "food": food,
        "hazards": hazards,
        "hazard_damage": int(ruleset.get("hazardDamagePerTurn") or 0),
    }


def flood_fill(
    width: int,
    height: int,
    start: tuple[int, int] | None,
    obstacles: set[tuple[int, int]],
) -> int:
    if not in_bounds(start, width, height) or start in obstacles:
        return 0
    queue = deque([start])
    visited = {start}
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS.values():
            nxt = (x + dx, y + dy)
            if not in_bounds(nxt, width, height):
                continue
            if nxt in obstacles or nxt in visited:
                continue
            visited.add(nxt)
            queue.append(nxt)
    return len(visited)


def count_open_neighbors(
    width: int,
    height: int,
    point: tuple[int, int] | None,
    obstacles: set[tuple[int, int]],
) -> int:
    if not in_bounds(point, width, height):
        return 0
    count = 0
    x, y = point
    for dx, dy in DIRECTIONS.values():
        nxt = (x + dx, y + dy)
        if in_bounds(nxt, width, height) and nxt not in obstacles:
            count += 1
    return count


def shortest_distance(
    width: int,
    height: int,
    start: tuple[int, int] | None,
    targets: set[tuple[int, int]],
    obstacles: set[tuple[int, int]],
) -> int | None:
    if not targets or not in_bounds(start, width, height):
        return None
    if start in targets:
        return 0
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        point, distance = queue.popleft()
        x, y = point
        for dx, dy in DIRECTIONS.values():
            nxt = (x + dx, y + dy)
            if not in_bounds(nxt, width, height):
                continue
            if nxt in obstacles or nxt in visited:
                continue
            if nxt in targets:
                return distance + 1
            visited.add(nxt)
            queue.append((nxt, distance + 1))
    return None


def evaluate_move(state: dict[str, Any], direction: str) -> MoveEval:
    context = build_context(state)
    width = context["width"]
    height = context["height"]
    you = context["you"]
    you_id = context["you_id"]
    you_head = point_tuple(you.get("head"))
    you_length = int(you.get("length") or 0)
    health = you.get("health") if isinstance(you.get("health"), int) else 100
    target = move_target(you_head, direction)
    food = context["food"]
    hits_food = target in food

    fatal_reasons: list[str] = []
    risk_tags: list[str] = []
    head_risk: list[str] = []
    head_attack: list[str] = []
    nearest_enemy_head_distance: int | None = None

    if not in_bounds(target, width, height):
        fatal_reasons.append("wall")
        risk_tags.append("fatal_wall")
        return MoveEval(
            direction=direction,
            target=target,
            fatal_reasons=fatal_reasons,
            risk_tags=risk_tags,
            reachable_space=0,
            open_neighbors=0,
            wall_distance=-1,
            food_distance=None,
            head_risk=head_risk,
            head_attack=head_attack,
            nearest_enemy_head_distance=nearest_enemy_head_distance,
            hits_food=hits_food,
        )

    solid_hits = context["solid_body_cells"].get(target, [])
    for hit in solid_hits:
        if hit["snake_id"] == you_id:
            fatal_reasons.append("self_body")
            risk_tags.append("fatal_self_body")
        else:
            fatal_reasons.append(f"opponent_body:{hit['snake_name']}")
            risk_tags.append("fatal_opponent_body")

    # Tails usually move away, but eating on a tail square means the tail does not vacate.
    if hits_food:
        for hit in context["tail_cells"].get(target, []):
            if hit["snake_id"] == you_id:
                fatal_reasons.append("self_tail_on_food")
                risk_tags.append("fatal_self_body")
            else:
                risk_tags.append("opponent_tail_on_food")

    if target in context["hazards"]:
        risk_tags.append("hazard")
        if health <= context["hazard_damage"] + 1 and not hits_food:
            fatal_reasons.append("hazard")
            risk_tags.append("fatal_hazard")

    if health <= 1 and not hits_food:
        fatal_reasons.append("out_of_health")
        risk_tags.append("fatal_out_of_health")

    for snake in context["active_snakes"]:
        snake_id = snake.get("id")
        if snake_id == you_id:
            continue
        head = point_tuple(snake.get("head"))
        if not in_bounds(head, width, height):
            continue
        distance = manhattan(target, head)
        if nearest_enemy_head_distance is None:
            nearest_enemy_head_distance = distance
        else:
            nearest_enemy_head_distance = min(nearest_enemy_head_distance, distance)
        for enemy_direction in DIRECTIONS:
            if move_target(head, enemy_direction) != target:
                continue
            enemy_name = snake_name(snake)
            enemy_length = int(snake.get("length") or 0)
            if enemy_length >= you_length:
                head_risk.append(enemy_name)
            else:
                head_attack.append(enemy_name)

    if head_risk:
        risk_tags.append("head_risk")
    if head_attack:
        risk_tags.append("head_attack_opportunity")
    if nearest_enemy_head_distance is not None and nearest_enemy_head_distance <= 1:
        risk_tags.append("adjacent_enemy_head")
    if hits_food:
        risk_tags.append("food")

    obstacles = set(context["obstacles"])
    if target in obstacles:
        reachable_space = 0
        open_neighbors = 0
        food_distance = None
    else:
        reachable_space = flood_fill(width, height, target, obstacles)
        open_neighbors = count_open_neighbors(width, height, target, obstacles)
        food_distance = shortest_distance(width, height, target, food, obstacles)

    x, y = target
    wall_distance = min(x, y, width - 1 - x, height - 1 - y)
    return MoveEval(
        direction=direction,
        target=target,
        fatal_reasons=fatal_reasons,
        risk_tags=dedupe(risk_tags),
        reachable_space=reachable_space,
        open_neighbors=open_neighbors,
        wall_distance=wall_distance,
        food_distance=food_distance,
        head_risk=dedupe(head_risk),
        head_attack=dedupe(head_attack),
        nearest_enemy_head_distance=nearest_enemy_head_distance,
        hits_food=hits_food,
    )


def move_score(eval_result: MoveEval, health: int) -> float:
    if eval_result.fatal_reasons:
        return -1_000_000.0 + eval_result.reachable_space
    score = (
        eval_result.reachable_space
        + eval_result.open_neighbors * 4.0
        + max(eval_result.wall_distance, 0) * 0.5
    )
    if eval_result.head_risk:
        score -= 30.0
    if eval_result.food_distance is not None:
        food_weight = 8.0
        if health <= 25:
            food_weight = 28.0
        elif health <= 45:
            food_weight = 18.0
        score += food_weight / (1.0 + eval_result.food_distance)
    return score


def analyze_move_entry(move_entry: dict[str, Any]) -> dict[str, Any] | None:
    state = move_entry.get("state")
    if not isinstance(state, dict):
        return None
    move = (move_entry.get("move") or {}).get("move")
    if move not in DIRECTIONS:
        return None

    context = build_context(state)
    width = context["width"]
    height = context["height"]
    you = context["you"]
    head = point_tuple(you.get("head"))
    health = you.get("health") if isinstance(you.get("health"), int) else 100
    length = int(you.get("length") or 0)
    board_area = max(1, width * height)

    evals = {direction: evaluate_move(state, direction) for direction in DIRECTIONS}
    chosen = evals[move]
    safe_evals = [eval_result for eval_result in evals.values() if not eval_result.fatal_reasons]
    best = max(evals.values(), key=lambda item: move_score(item, health))
    best_safe = max(safe_evals, key=lambda item: move_score(item, health)) if safe_evals else None

    obstacles = set(context["obstacles"])
    if head in obstacles:
        obstacles.remove(head)
    before_food_distance = shortest_distance(width, height, head, context["food"], obstacles)

    tags = list(chosen.risk_tags)
    if chosen.reachable_space < max(1, length):
        tags.append("space_less_than_length")
    elif chosen.reachable_space < max(10, int(length * 1.5)):
        tags.append("low_space")
    if chosen.reachable_space / board_area < 0.12:
        tags.append("tiny_region")
    if chosen.open_neighbors <= 1:
        tags.append("dead_end")
    elif chosen.open_neighbors == 2:
        tags.append("corridor")
    if chosen.wall_distance == 0:
        tags.append("edge")

    if health <= 25:
        if chosen.food_distance is None:
            tags.append("no_food_path_critical")
        elif (
            before_food_distance is not None
            and chosen.food_distance >= before_food_distance
            and not chosen.hits_food
        ):
            tags.append("no_food_progress_critical")
    elif health <= 45:
        if (
            chosen.food_distance is None
            or (
                before_food_distance is not None
                and chosen.food_distance >= before_food_distance
                and not chosen.hits_food
            )
        ):
            tags.append("low_health_no_food_progress")

    if best_safe is not None:
        if chosen.fatal_reasons:
            tags.append("safer_alternative_available")
        if chosen.head_risk and not best_safe.head_risk:
            tags.append("head_safe_alternative_available")
        space_margin = max(8, length // 2)
        if (
            not chosen.fatal_reasons
            and best_safe.reachable_space >= chosen.reachable_space + space_margin
        ):
            tags.append("missed_space")
        if chosen.open_neighbors <= 1 and best_safe.open_neighbors >= 2:
            tags.append("dead_end_alternative_available")

    timeout = (((state.get("game") or {}).get("timeout")) or 500)
    latency = move_entry.get("latency_ms")
    if isinstance(latency, (int, float)):
        if latency >= timeout:
            tags.append("timeout_risk")
        elif latency >= timeout * 0.8:
            tags.append("high_latency")

    return {
        "chosen": chosen,
        "best": best,
        "best_safe": best_safe,
        "evals": evals,
        "tags": dedupe(tags),
        "before_food_distance": before_food_distance,
    }


def find_death_move(data: dict[str, Any], loss_turn: int | None) -> dict[str, Any] | None:
    moves = data.get("moves") or []
    if not moves:
        return None
    if isinstance(loss_turn, int):
        preferred_turns = [loss_turn - 1, loss_turn]
        for turn in preferred_turns:
            for move in reversed(moves):
                if move.get("turn") == turn:
                    return move
        for move in reversed(moves):
            turn = move.get("turn")
            if isinstance(turn, int) and turn <= loss_turn:
                return move
    return moves[-1]


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def format_point(point: tuple[int, int] | None) -> str:
    if point is None:
        return "?"
    return f"({point[0]},{point[1]})"


def eval_to_dict(eval_result: MoveEval | None) -> dict[str, Any] | None:
    if eval_result is None:
        return None
    return {
        "direction": eval_result.direction,
        "target": list(eval_result.target) if eval_result.target is not None else None,
        "fatal_reasons": eval_result.fatal_reasons,
        "risk_tags": eval_result.risk_tags,
        "reachable_space": eval_result.reachable_space,
        "open_neighbors": eval_result.open_neighbors,
        "wall_distance": eval_result.wall_distance,
        "food_distance": eval_result.food_distance,
        "head_risk": eval_result.head_risk,
        "head_attack": eval_result.head_attack,
        "nearest_enemy_head_distance": eval_result.nearest_enemy_head_distance,
        "hits_food": eval_result.hits_food,
    }


def label_counter(counter: Counter) -> list[tuple[str, int]]:
    return [(CAUSE_LABELS.get(key, TAG_LABELS.get(key, key)), value) for key, value in counter.most_common()]


def short_tags(tags: list[str], limit: int = 5) -> str:
    labels = []
    for tag in tags:
        if tag.startswith("loss:"):
            cause = tag.split(":", 1)[1]
            labels.append(f"Loss: {CAUSE_LABELS.get(cause, cause)}")
        else:
            labels.append(TAG_LABELS.get(tag, tag))
    if len(labels) > limit:
        labels = labels[:limit] + [f"+{len(tags) - limit}"]
    return ", ".join(labels) if labels else "-"


def analyze_log(path: Path, data: dict[str, Any]) -> dict[str, Any] | None:
    you_id = infer_you_id(data)
    if you_id is None:
        return None

    profiles, opponent_names = collect_opponent_profiles(data, you_id)
    id_to_name = {snake_id: profile["name"] for snake_id, profile in profiles.items()}

    outcome = infer_outcome(data, you_id)
    turn = infer_game_turn(data)
    you_result = result_snake(data, you_id)
    event = (you_result or {}).get("elimination_event") if you_result else None
    cause = event.get("cause") if isinstance(event, dict) else None
    loss_turn = event.get("turn") if isinstance(event, dict) and isinstance(event.get("turn"), int) else None
    killer_id = event.get("by") if isinstance(event, dict) else None
    if killer_id == you_id:
        killer_name = "self"
    else:
        killer_name = id_to_name.get(killer_id, killer_id) if killer_id else None

    move_risks: Counter[str] = Counter()
    move_count = 0
    for move_entry in data.get("moves") or []:
        move_analysis = analyze_move_entry(move_entry)
        if move_analysis is None:
            continue
        move_count += 1
        move_risks.update(move_analysis["tags"])

    final_move = find_death_move(data, loss_turn) if outcome == "loss" else None
    final_analysis = analyze_move_entry(final_move) if final_move is not None else None
    final_tags: list[str] = []
    final_eval: MoveEval | None = None
    best_safe: MoveEval | None = None
    if final_analysis is not None:
        final_tags = list(final_analysis["tags"])
        final_eval = final_analysis["chosen"]
        best_safe = final_analysis["best_safe"]

    if cause:
        final_tags.append(f"loss:{cause}")

    return {
        "path": str(path),
        "file": path.name,
        "game_id": data.get("game_id") or path.stem,
        "outcome": outcome,
        "turn": turn,
        "you_id": you_id,
        "you_name": (you_result or {}).get("name") if you_result else None,
        "move_count": len(data.get("moves") or []),
        "analyzed_move_count": move_count,
        "opponents": sorted(opponent_names),
        "opponent_profiles": profiles,
        "cause": cause,
        "loss_turn": loss_turn,
        "killer_id": killer_id,
        "killer_name": killer_name,
        "move_risks": move_risks,
        "final_tags": dedupe(final_tags),
        "final_move_turn": final_move.get("turn") if final_move else None,
        "final_move": (final_move.get("move") or {}).get("move") if final_move else None,
        "final_eval": final_eval,
        "best_safe": best_safe,
    }


def add_opponent_stats(
    opponent_stats: dict[str, dict[str, Any]],
    game: dict[str, Any],
) -> None:
    for opponent in game["opponents"]:
        stats = opponent_stats.setdefault(
            opponent,
            {
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "unknown": 0,
                "loss_causes": Counter(),
                "kills": Counter(),
                "risk_tags": Counter(),
                "turns": [],
            },
        )
        stats["games"] += 1
        stats["turns"].append(game["turn"])
        outcome = game["outcome"]
        if outcome == "win":
            stats["wins"] += 1
        elif outcome == "loss":
            stats["losses"] += 1
            if game["cause"]:
                stats["loss_causes"][game["cause"]] += 1
            stats["risk_tags"].update(game["final_tags"])
            if game["killer_name"] == opponent:
                stats["kills"][game["cause"] or "unknown"] += 1
        elif outcome == "draw":
            stats["draws"] += 1
        else:
            stats["unknown"] += 1


def summarize(games: list[dict[str, Any]], duplicate_skips: int) -> dict[str, Any]:
    outcomes = Counter(game["outcome"] for game in games)
    loss_causes = Counter(game["cause"] or "unknown" for game in games if game["outcome"] == "loss")
    final_loss_tags = Counter()
    risk_by_outcome: dict[str, Counter[str]] = defaultdict(Counter)
    turns_by_outcome: dict[str, list[int]] = defaultdict(list)
    move_counts_by_outcome: dict[str, int] = Counter()
    opponent_stats: dict[str, dict[str, Any]] = {}

    for game in games:
        outcome = game["outcome"]
        if isinstance(game["turn"], int):
            turns_by_outcome[outcome].append(game["turn"])
        move_counts_by_outcome[outcome] += int(game["analyzed_move_count"] or 0)
        risk_by_outcome[outcome].update(game["move_risks"])
        if outcome == "loss":
            final_loss_tags.update(game["final_tags"])
        add_opponent_stats(opponent_stats, game)

    return {
        "games": games,
        "duplicate_skips": duplicate_skips,
        "outcomes": outcomes,
        "loss_causes": loss_causes,
        "final_loss_tags": final_loss_tags,
        "risk_by_outcome": risk_by_outcome,
        "turns_by_outcome": turns_by_outcome,
        "move_counts_by_outcome": move_counts_by_outcome,
        "opponent_stats": opponent_stats,
    }


def fmt_percent(part: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{part / total * 100:.1f}%"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in str_rows))
        for index in range(len(headers))
    ]
    lines = ["  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    for row in str_rows:
        lines.append("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))
    return "\n".join(lines)


def mean_or_dash(values: list[int | None]) -> str:
    clean = [value for value in values if isinstance(value, int)]
    if not clean:
        return "-"
    return f"{mean(clean):.1f}"


def median_or_dash(values: list[int | None]) -> str:
    clean = [value for value in values if isinstance(value, int)]
    if not clean:
        return "-"
    return f"{median(clean):.1f}"


def build_recommendations(summary: dict[str, Any], top_n: int) -> list[str]:
    recommendations: list[str] = []
    causes: Counter = summary["loss_causes"]
    final_tags: Counter = summary["final_loss_tags"]
    total_losses = sum(causes.values())

    def add(text: str) -> None:
        if text not in recommendations:
            recommendations.append(text)

    if causes["snake-self-collision"]:
        add(
            "Selbstkollisionen: vor dem Strategie-Voting harte Veto-Regel fuer eigene Koerperzellen "
            "und fuer Zuege mit Flood-Fill < eigene Laenge einbauen."
        )
    if causes["head-collision"]:
        add(
            "Head-to-head: alle Felder, die gleich lange, laengere oder wegen Blackout-Laenge unsichere Gegner "
            "im naechsten Zug erreichen koennen, deutlich staerker sperren; Angriffe nur gegen klar kuerzere "
            "Gegner erlauben."
        )
    if causes["snake-collision"]:
        add(
            "Gegnerkoerper: sichtbare Gegnersegmente und zuletzt gesehene Blackout-Spuren laenger als "
            "Unsicherheitszonen behandeln, besonders wenn der Gegner aus dem Sichtfeld verschwindet."
        )
    if causes["wall-collision"]:
        add(
            "Wandkollisionen: Out-of-bounds-Zuege komplett vor allen Scores herausfiltern und am Rand "
            "einen Escape-Bonus fuer Felder mit mehreren Ausgaengen geben."
        )
    if causes["out-of-health"]:
        add(
            "Health: Food-Ziele frueher committen, wenn Health < 45 ist; bei Health < 25 nur noch "
            "Zuege erlauben, die Food-Distanz senken oder direkt fressen."
        )
    if final_tags["space_less_than_length"] or final_tags["low_space"] or final_tags["tiny_region"]:
        add(
            "Raumplanung: nicht nur den besten Einzelzug bewerten, sondern Moves mit kleinem erreichbaren "
            "Gebiet als Endspiel-Falle markieren, auch wenn sie kurzfristig sicher aussehen."
        )
    if final_tags["dead_end"] or final_tags["dead_end_alternative_available"]:
        add(
            "Sackgassen: Zuege mit nur einem offenen Nachbarn stark abwerten, ausser der Tail-Chase ist "
            "simuliert erreichbar und der naechste Zug bleibt offen."
        )
    if final_tags["missed_space"] or final_tags["safer_alternative_available"]:
        add(
            "Safety-Override: nach der Strategieauswahl einen finalen Vergleich gegen die beste sichere "
            "Alternative machen und bei grossem Raumvorteil automatisch wechseln."
        )
    if final_tags["head_safe_alternative_available"]:
        add(
            "Kopf-Risiko: bei Head-Risk und vorhandener kopfsicherer Alternative den riskanten Zug nur "
            "nehmen, wenn dadurch sicher ein kuerzerer Gegner eliminiert wird."
        )

    opponent_rows = opponent_problem_rows(summary, top_n=top_n)
    if opponent_rows:
        worst = ", ".join(row[0] for row in opponent_rows[:3])
        add(f"Gegnerspezifisch: zuerst Profile gegen {worst} anschauen; dort ist die Loss-Rate am auffaelligsten.")

    if not recommendations and total_losses == 0:
        add("Keine Verluste in den ausgewaehlten Logs gefunden. Naechster Schritt: mehr Spiele oder schwierigere Gegner loggen.")
    elif not recommendations:
        add("Die Verluste sind breit gestreut. Mehr Logs sammeln oder mit --detail einzelne Endstellungen ansehen.")

    return recommendations[:top_n]


def opponent_problem_rows(summary: dict[str, Any], top_n: int) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for name, stats in summary["opponent_stats"].items():
        games = stats["games"]
        if games == 0:
            continue
        losses = stats["losses"]
        if losses == 0:
            continue
        loss_rate = losses / games
        causes = ", ".join(
            f"{CAUSE_LABELS.get(cause, cause)}:{count}"
            for cause, count in stats["loss_causes"].most_common(2)
        )
        kills = sum(stats["kills"].values())
        turns = [turn for turn in stats["turns"] if isinstance(turn, int)]
        avg_turn = f"{mean(turns):.1f}" if turns else "-"
        rows.append(
            [
                name,
                games,
                f"{loss_rate * 100:.1f}%",
                losses,
                stats["wins"],
                kills,
                avg_turn,
                causes or "-",
            ]
        )
    rows.sort(key=lambda row: (float(row[2].rstrip("%")), row[1]), reverse=True)
    return rows[:top_n]


def print_report(summary: dict[str, Any], top_n: int, detail: int) -> None:
    games = summary["games"]
    outcomes: Counter = summary["outcomes"]
    total = len(games)
    full_histories = sum(1 for game in games if game["move_count"])
    turns = [game["turn"] for game in games if isinstance(game["turn"], int)]

    print("\n=== Game-Log-Analyse ===")
    print(f"Spiele analysiert: {total} ({full_histories} mit Move-Historie)")
    if summary["duplicate_skips"]:
        print(f"Duplikate uebersprungen: {summary['duplicate_skips']} (reichhaltigstes Log je game_id bevorzugt)")
    print(
        "Outcomes: "
        f"Wins {outcomes['win']} ({fmt_percent(outcomes['win'], total)}), "
        f"Losses {outcomes['loss']} ({fmt_percent(outcomes['loss'], total)}), "
        f"Draws {outcomes['draw']} ({fmt_percent(outcomes['draw'], total)}), "
        f"Unknown {outcomes['unknown']}"
    )
    print(f"Turns: avg {mean_or_dash(turns)}, median {median_or_dash(turns)}")

    print("\n=== Verlustgruende ===")
    cause_rows = [
        [CAUSE_LABELS.get(cause, cause), count, fmt_percent(count, max(1, outcomes["loss"]))]
        for cause, count in summary["loss_causes"].most_common()
    ]
    print(table(["Grund", "Anzahl", "Anteil Losses"], cause_rows) if cause_rows else "Keine Verluste gefunden.")

    print("\n=== Muster im Todeszug ===")
    tag_rows = [
        [TAG_LABELS.get(tag, tag), count, fmt_percent(count, max(1, outcomes["loss"]))]
        for tag, count in summary["final_loss_tags"].most_common(top_n)
        if not tag.startswith("loss:")
    ]
    print(table(["Muster", "Anzahl", "Anteil Losses"], tag_rows) if tag_rows else "Keine Todeszug-Muster verfuegbar.")

    print("\n=== Risiko-Muster pro 100 Moves ===")
    rows: list[list[Any]] = []
    for outcome in ("loss", "win", "draw"):
        moves = summary["move_counts_by_outcome"][outcome]
        if not moves:
            continue
        for tag, count in summary["risk_by_outcome"][outcome].most_common(5):
            rows.append([outcome, TAG_LABELS.get(tag, tag), f"{count / moves * 100:.2f}", count, moves])
    print(table(["Outcome", "Muster", "pro 100 Moves", "Anzahl", "Moves"], rows) if rows else "Keine Move-Historien auswertbar.")

    print("\n=== Gegnerprofile ===")
    opponent_rows = opponent_problem_rows(summary, top_n=top_n)
    if opponent_rows:
        print(table(["Gegner", "Games", "Loss-Rate", "Losses", "Wins", "Kills", "Avg Turn", "Top Causes"], opponent_rows))
    else:
        print("Keine auffaelligen Gegnerprofile mit Losses.")

    print("\n=== Verbesserungsvorschlaege ===")
    for index, recommendation in enumerate(build_recommendations(summary, top_n=top_n), start=1):
        print(f"{index}. {recommendation}")

    print("\n=== Beispielverluste ===")
    losses = [game for game in games if game["outcome"] == "loss"]
    losses.sort(key=lambda game: (game["file"], game["loss_turn"] or -1), reverse=True)
    for game in losses[:detail]:
        cause = CAUSE_LABELS.get(game["cause"], game["cause"] or "unknown")
        killer = f" gegen {game['killer_name']}" if game["killer_name"] else ""
        move = game["final_move"] or "?"
        final_eval = game["final_eval"]
        best_safe = game["best_safe"]
        print(f"- {game['file']} | Turn {game['loss_turn'] or game['turn']} | {cause}{killer} | Move {move}")
        if final_eval is not None:
            print(
                "  Gewaehlt: "
                f"{final_eval.direction} -> {format_point(final_eval.target)}, "
                f"space={final_eval.reachable_space}, exits={final_eval.open_neighbors}, "
                f"food={final_eval.food_distance if final_eval.food_distance is not None else '-'}, "
                f"tags={short_tags(game['final_tags'])}"
            )
        else:
            print("  Kein Todeszug im Log vorhanden; nur Endzustand auswertbar.")
        if best_safe is not None and final_eval is not None and best_safe.direction != final_eval.direction:
            print(
                "  Beste sichere Alternative: "
                f"{best_safe.direction} -> {format_point(best_safe.target)}, "
                f"space={best_safe.reachable_space}, exits={best_safe.open_neighbors}, "
                f"food={best_safe.food_distance if best_safe.food_distance is not None else '-'}"
            )


def counter_to_plain(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.items()}


def plain_summary(summary: dict[str, Any]) -> dict[str, Any]:
    games = []
    for game in summary["games"]:
        games.append(
            {
                "path": game["path"],
                "file": game["file"],
                "game_id": game["game_id"],
                "outcome": game["outcome"],
                "turn": game["turn"],
                "move_count": game["move_count"],
                "opponents": game["opponents"],
                "cause": game["cause"],
                "loss_turn": game["loss_turn"],
                "killer_name": game["killer_name"],
                "move_risks": counter_to_plain(game["move_risks"]),
                "final_tags": game["final_tags"],
                "final_move": game["final_move"],
                "final_eval": eval_to_dict(game["final_eval"]),
                "best_safe": eval_to_dict(game["best_safe"]),
            }
        )

    opponents = {}
    for name, stats in summary["opponent_stats"].items():
        turns = [turn for turn in stats["turns"] if isinstance(turn, int)]
        opponents[name] = {
            "games": stats["games"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "draws": stats["draws"],
            "unknown": stats["unknown"],
            "loss_rate": stats["losses"] / stats["games"] if stats["games"] else 0,
            "avg_turn": mean(turns) if turns else None,
            "loss_causes": counter_to_plain(stats["loss_causes"]),
            "kills": counter_to_plain(stats["kills"]),
            "risk_tags": counter_to_plain(stats["risk_tags"]),
        }

    return {
        "games_analyzed": len(games),
        "duplicate_skips": summary["duplicate_skips"],
        "outcomes": counter_to_plain(summary["outcomes"]),
        "loss_causes": counter_to_plain(summary["loss_causes"]),
        "final_loss_tags": counter_to_plain(summary["final_loss_tags"]),
        "opponents": opponents,
        "games": games,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analysiert Battlesnake game_logs und erklaert Verlustmuster, Gegnerprofile und Verbesserungen."
    )
    parser.add_argument("log_dir", nargs="?", default="game_logs", type=Path, help="Ordner mit JSON-Logs")
    parser.add_argument("--limit", type=int, default=None, help="Nur die neuesten N JSON-Dateien betrachten")
    parser.add_argument("--top", type=int, default=10, help="Anzahl Top-Zeilen pro Report-Sektion")
    parser.add_argument("--detail", type=int, default=8, help="Anzahl konkreter Beispielverluste")
    parser.add_argument("--include-duplicates", action="store_true", help="Duplikate mit gleicher game_id nicht deduplizieren")
    parser.add_argument("--json-out", type=Path, default=None, help="Maschinenlesbare Zusammenfassung als JSON schreiben")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.log_dir.exists():
        raise SystemExit(f"Log-Ordner nicht gefunden: {args.log_dir}")

    paths = sorted(args.log_dir.glob("*.json"), key=lambda path: path.name)
    if args.limit is not None:
        paths = paths[-args.limit :]
    selected_paths, duplicate_skips = choose_log_paths(paths, args.include_duplicates)

    games: list[dict[str, Any]] = []
    for path in selected_paths:
        data = load_json(path)
        if data is None:
            continue
        analysis = analyze_log(path, data)
        if analysis is not None:
            games.append(analysis)

    summary = summarize(games, duplicate_skips)
    print_report(summary, top_n=max(1, args.top), detail=max(0, args.detail))

    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(plain_summary(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\nJSON geschrieben: {args.json_out}")


if __name__ == "__main__":
    main()
