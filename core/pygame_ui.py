import os
import pygame
from typing import Optional
from core.env.core import SnakeEnv
from core.env.types import DIR_OFFSETS, GridType

BG_COLOR = (15, 23, 42)
GRID_COLOR = (30, 41, 59)
TEXT_COLOR = (248, 250, 252)
SIDEBAR_BG = (22, 28, 42)
MCTS_SIDEBAR_W = 420
DEFAULT_BODY_COLOR = (52, 211, 153)
APPLE_COLOR = (244, 63, 94)
OBST_COLOR = (100, 116, 139)
SEGMENT_MARGIN = 2
TAIL_ALPHA = 0.5
JETBRAINS_FONT_SIZE = 16


class PygameUI:
    def __init__(
        self,
        cell_size: int = 40,
        fps: int = 30,
        agent_color: Optional[tuple[int, int, int]] = None,
    ):
        self.cell_size = cell_size
        self.fps = fps
        self.agent_color = agent_color if agent_color else DEFAULT_BODY_COLOR
        self.screen = None
        self.clock = None
        self.paused = False
        self.sidebar_w = 0

    def init_screen(self, width: int, height: int, sidebar_w: int = 0):
        if not pygame.get_init():
            pygame.init()

        self.metrics_height = 40
        self.sidebar_w = sidebar_w
        grid_w = width * self.cell_size
        self.screen_width = grid_w + self.sidebar_w
        self.screen_height = height * self.cell_size + self.metrics_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        pygame.display.set_caption("Snake RL")
        pygame.display.set_icon(pygame.Surface((1, 1)))
        self.clock = pygame.time.Clock()

        font_path = os.path.join(os.path.dirname(__file__), "../resources/JetBrainsMono-Regular.ttf")
        try:
            self.font = pygame.font.Font(font_path, JETBRAINS_FONT_SIZE)
            self.font_small = pygame.font.Font(font_path, 13)
        except Exception:
            self.font = pygame.font.SysFont(None, 24)
            self.font_small = pygame.font.SysFont(None, 14)

        obs_size = self.cell_size - 2
        self.obst_surf = pygame.Surface((obs_size, obs_size))
        self.obst_surf.fill((20, 30, 50))
        pygame.draw.rect(self.obst_surf, OBST_COLOR, (0, 0, obs_size, obs_size), 1)
        for i in range(-obs_size, obs_size * 2, 8):
            pygame.draw.line(self.obst_surf, OBST_COLOR, (i, 0), (i + obs_size, obs_size), 3)

    def confirm_exit(self) -> bool:
        if not getattr(self, "screen", None):
            return True

        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))

        text_surf = self.font.render("Exit game? (Y/N)", True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        overlay.blit(text_surf, text_rect)

        self.screen.blit(overlay, (0, 0))  # pyright: ignore[reportOptionalMemberAccess]
        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        return True
                    if event.key in (pygame.K_n, pygame.K_ESCAPE):
                        return False
            self.clock.tick(self.fps)  # pyright: ignore[reportOptionalMemberAccess]

    def handle_events(self, env: SnakeEnv) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT and self.confirm_exit():
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION) and self.paused:
                b1, _, b3 = pygame.mouse.get_pressed()
                if b1 or b3:
                    c, r = (
                        event.pos[0] // self.cell_size,
                        event.pos[1] // self.cell_size,
                    )
                    grid_type = GridType.APPLE if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else GridType.OBSTACLE
                    env.set_item_dyn((r, c), grid_type, b1)
        return True

    def _draw_grid(self, screen, env):
        for r in range(env.height):
            for c in range(env.width):
                pygame.draw.rect(
                    screen,
                    GRID_COLOR,
                    (
                        c * self.cell_size,
                        r * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                    1,
                )

    def _draw_obstacles(self, screen, env):
        for r, c in env.obstacles:
            screen.blit(self.obst_surf, (c * self.cell_size + 1, r * self.cell_size + 1))

    def _draw_apples(self, screen, env):
        time_ms = pygame.time.get_ticks()
        ripple_progress = (time_ms % 1000) / 1000.0
        base_hs = self.cell_size // 4
        ripple_hs = int(base_hs + ripple_progress * (self.cell_size // 2))
        alpha = int(255 * (1.0 - ripple_progress))

        for r, c in env.apples:
            cx, cy = (
                c * self.cell_size + self.cell_size // 2,
                r * self.cell_size + self.cell_size // 2,
            )
            pygame.draw.polygon(
                screen,
                APPLE_COLOR,
                [
                    (cx, cy - base_hs),
                    (cx + base_hs, cy),
                    (cx, cy + base_hs),
                    (cx - base_hs, cy),
                ],
            )

            ripple_surf = pygame.Surface((self.cell_size * 2, self.cell_size * 2), pygame.SRCALPHA)
            pygame.draw.polygon(
                ripple_surf,
                (*APPLE_COLOR, alpha),
                [
                    (self.cell_size, self.cell_size - ripple_hs),
                    (self.cell_size + ripple_hs, self.cell_size),
                    (self.cell_size, self.cell_size + ripple_hs),
                    (self.cell_size - ripple_hs, self.cell_size),
                ],
            )
            screen.blit(ripple_surf, (cx - self.cell_size, cy - self.cell_size))

    def _draw_snake(self, screen, env):
        snake = env.snake
        if not snake.alive:
            return

        slen = len(snake.body)
        color = self.agent_color
        w = h = self.cell_size - SEGMENT_MARGIN * 2

        for i in reversed(range(slen)):
            r, c = snake.body[i]
            alpha = int(255 * (1.0 - (i / max(1, slen - 1)) * (1.0 - TAIL_ALPHA)))
            x, y = (
                c * self.cell_size + SEGMENT_MARGIN,
                r * self.cell_size + SEGMENT_MARGIN,
            )

            seg_surf = pygame.Surface((w, h), pygame.SRCALPHA)
            seg_surf.fill((*color, alpha))

            if i == 0:
                cx, cy, s = w / 2.0, h / 2.0, w / 3.0
                dy, dx = DIR_OFFSETS[snake.dir]
                pts = [
                    (cx + dx * (s / 2.0), cy + dy * (s / 2.0)),
                    (cx - dx * (s / 2.0) - dy * s, cy - dy * (s / 2.0) + dx * s),
                    (cx - dx * (s / 2.0) + dy * s, cy - dy * (s / 2.0) - dx * s),
                ]
                pygame.draw.polygon(seg_surf, (*BG_COLOR, alpha), pts)

            screen.blit(seg_surf, (x, y))

            if i > 0:
                self._draw_snake_connections(screen, r, c, snake.body[i - 1], x, y, w, h, color, alpha)

    def _draw_snake_connections(self, screen, r, c, prev_pos, x, y, w, h, color, alpha):
        pr, pc = prev_pos
        if abs(pr - r) + abs(pc - c) == 1:
            gx, gy, gw, gh = x, y, w, h
            if pc > c:
                gx, gw = x + w, SEGMENT_MARGIN * 2
            elif pc < c:
                gx, gw = x - SEGMENT_MARGIN * 2, SEGMENT_MARGIN * 2
            elif pr > r:
                gy, gh = y + h, SEGMENT_MARGIN * 2
            elif pr < r:
                gy, gh = y - SEGMENT_MARGIN * 2, SEGMENT_MARGIN * 2

            gap_surf = pygame.Surface((gw, gh), pygame.SRCALPHA)
            gap_surf.fill((*color, alpha))
            screen.blit(gap_surf, (gx, gy))

    def _draw_mcts_sidebar(self, screen, env: SnakeEnv):
        lines = getattr(env, "mcts_panel", None)
        if not lines:
            return
        grid_w = env.width * self.cell_size
        h = env.height * self.cell_size
        pygame.draw.rect(screen, SIDEBAR_BG, (grid_w, 0, self.sidebar_w, h))
        x0 = grid_w + 6
        y = 8
        fs = self.font_small
        line_h = fs.get_height() + 2
        for i, line in enumerate(lines):
            surf = fs.render(line, True, TEXT_COLOR)
            screen.blit(surf, (x0, y + i * line_h))

    def _resize_for_panel(self, env: SnakeEnv) -> None:
        panel = getattr(env, "mcts_panel", None)
        want = MCTS_SIDEBAR_W if panel else 0
        grid_w = env.width * self.cell_size
        grid_h = env.height * self.cell_size
        need_w = grid_w + want
        need_h = grid_h + self.metrics_height
        if self.screen is not None and (need_w, need_h) == (self.screen_width, self.screen_height):
            self.sidebar_w = want
            return
        self.sidebar_w = want
        self.screen_width = need_w
        self.screen_height = need_h
        if not pygame.get_init():
            pygame.init()
        self.screen = pygame.display.set_mode((need_w, need_h))
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def _draw_metrics(self, screen, env, total_rewards):
        metrics_y = env.height * self.cell_size
        pygame.draw.rect(screen, GRID_COLOR, (0, metrics_y, self.screen_width, self.metrics_height))

        snake, reward = env.snake, total_rewards or 0.0
        stats = f"Steps: {env.step_count} | Snake Length: {len(snake.body)} | Reward: {reward:.1f}"
        if not snake.alive:
            stats = f"DEAD | Reward: {reward:.1f}"

        prefix_surf = self.font.render("Agent: ", True, self.agent_color)
        screen.blit(prefix_surf, (10, metrics_y + 10))
        screen.blit(
            self.font.render(stats, True, TEXT_COLOR),
            (10 + prefix_surf.get_width(), metrics_y + 10),
        )

    def render(self, env: SnakeEnv, total_rewards: Optional[float] = None):
        if getattr(self, "screen", None) is None:
            panel = getattr(env, "mcts_panel", None)
            sw = MCTS_SIDEBAR_W if panel else 0
            self.init_screen(env.width, env.height, sidebar_w=sw)
        else:
            self._resize_for_panel(env)

        screen = self.screen
        assert screen is not None

        screen.fill(BG_COLOR)

        self._draw_grid(screen, env)
        self._draw_mcts_sidebar(screen, env)
        self._draw_obstacles(screen, env)
        self._draw_apples(screen, env)
        self._draw_snake(screen, env)

        if self.paused:
            screen.blit(
                self.font.render(
                    "PAUSED (Obstacle (Shift to Apple), L: Add, R: Remove, Space: Resume)",
                    True,
                    TEXT_COLOR,
                ),
                (10, 10),
            )

        self._draw_metrics(screen, env, total_rewards)

        pygame.display.flip()
        self.clock.tick(self.fps)  # pyright: ignore[reportOptionalMemberAccess]

    def quit(self):
        if pygame.get_init():
            pygame.quit()
