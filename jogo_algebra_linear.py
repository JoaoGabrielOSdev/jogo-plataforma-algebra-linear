import math
import sys
import numpy as np
import pygame

# ============================================================
# JOGO DE PLATAFORMA 2D
# Álgebra Linear aplicada a Jogos
#
# Dependências:
#   pip install pygame numpy
#
# Controles:
#   A / D ou setas: mover
#   W / Espaço: pular
#   R: reiniciar
#   ESC: sair
#
# Conceitos de Álgebra Linear presentes:
#   - vetores em R^2
#   - base canônica
#   - soma de vetores
#   - multiplicação por escalar
#   - norma e distância
#   - normalização
#   - produto interno
#   - matriz de rotação
#   - transformação linear
#
# MELHORIAS ADICIONADAS (marcadas com [MELHORIA] no código):
#   1. Projéteis: o inimigo atira na direção do jogador usando o
#      vetor unitário direction_unit (multiplicação por escalar).
#   2. Reflexão: os projéteis quicam nas plataformas usando a
#      fórmula v' = v - 2 (v . n) n  (produto interno).
#   3. Cone de visão real: as bordas do campo de visão são obtidas
#      girando o vetor forward com a matriz de rotação R(+-40°).
#   4. Varredura do olhar: na patrulha, o inimigo "olha" para cima
#      e para baixo girando forward com R(theta(t)).
#   5. HUD mostra o ângulo entre forward e o jogador, obtido por
#      theta = arccos(forward . direction_unit).
# ============================================================

pygame.init()

WIDTH, HEIGHT = 1000, 600
FPS = 60

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Álgebra Linear - Platform Hunter")
clock = pygame.time.Clock()

font = pygame.font.Font(None, 28)
small_font = pygame.font.Font(None, 22)
title_font = pygame.font.Font(None, 36)

# ------------------------------------------------------------
# Mundo
# ------------------------------------------------------------

WORLD_WIDTH = 2400
WORLD_HEIGHT = 900
GRAVITY = 1200.0

# Base canônica de R^2:
E1 = np.array([1.0, 0.0])
E2 = np.array([0.0, 1.0])

# ------------------------------------------------------------
# Plataformas
# Cada plataforma: x, y, largura, altura
# ------------------------------------------------------------

platforms = [
    pygame.Rect(0, 540, 700, 60),
    pygame.Rect(780, 470, 260, 40),
    pygame.Rect(1120, 390, 260, 40),
    pygame.Rect(1450, 500, 300, 40),
    pygame.Rect(1780, 410, 280, 40),
    pygame.Rect(2120, 520, 280, 60),
    pygame.Rect(900, 650, 300, 35),
    pygame.Rect(1320, 700, 280, 35),
]

goal_rect = pygame.Rect(2320, 440, 30, 80)

# ------------------------------------------------------------
# Cores
# ------------------------------------------------------------

BG = (25, 30, 42)
PLATFORM_COLOR = (85, 105, 130)
PLAYER_COLOR = (70, 180, 255)
ENEMY_COLOR = (235, 80, 90)
TEXT_COLOR = (235, 240, 245)
VISION_COLOR = (240, 190, 70)
GOAL_COLOR = (90, 220, 120)

# ------------------------------------------------------------
# Funções de Álgebra Linear
# ------------------------------------------------------------

def normalize(v):
    """Retorna v / ||v||. Para vetor nulo, retorna o próprio vetor."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.copy()
    return v / norm


def rotation_matrix(theta):
    """Matriz de rotação 2D."""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([
        [c, -s],
        [s,  c]
    ], dtype=float)


def transform_points(points, position, angle, scale=1.0):
    """
    Aplica escala, rotação e translação aos pontos locais.
    A transformação linear é feita por matrizes; a translação
    é aplicada depois.
    """
    S = np.array([
        [scale, 0.0],
        [0.0, scale]
    ])

    R = rotation_matrix(angle)
    A = R @ S

    result = []
    for point in points:
        p = np.array(point, dtype=float)
        world = A @ p + position
        result.append(tuple(world))
    return result


# [MELHORIA] Reflexão de um vetor em relação a uma superfície.
def reflect(v, n):
    """
    Reflete o vetor v em uma superfície de normal unitária n:
        v' = v - 2 (v . n) n
    (v . n) n é a projeção de v sobre n; subtraí-la duas vezes
    inverte só a componente perpendicular à superfície.
    """
    return v - 2.0 * np.dot(v, n) * n


# [MELHORIA] Normal da face de um retângulo atingida por um ponto.
def collision_normal(previous_pos, rect):
    """
    Usa a posição anterior do ponto para descobrir por qual face
    ele entrou no retângulo. Retorna um vetor da base canônica
    (ou seu oposto): +-E1 para paredes laterais, +-E2 para topo/fundo.
    """
    x, y = previous_pos
    if x < rect.left:
        return -E1
    if x > rect.right:
        return E1
    if y < rect.top:
        return -E2
    return E2


# ------------------------------------------------------------
# Classe Jogador
# ------------------------------------------------------------

class Player:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pos = np.array([120.0, 450.0], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.width = 36
        self.height = 50
        self.speed = 260.0
        self.jump_speed = -600.0
        self.on_ground = False

    @property
    def rect(self):
        return pygame.Rect(
            int(self.pos[0]),
            int(self.pos[1]),
            self.width,
            self.height
        )

    @property
    def center(self):
        return self.pos + np.array([
            self.width / 2,
            self.height / 2
        ])

    def update(self, dt, keys):
        # Vetor de movimento horizontal.
        move = 0.0

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            move -= 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            move += 1.0

        self.velocity[0] = move * self.speed

        if (keys[pygame.K_w] or keys[pygame.K_SPACE]
                or keys[pygame.K_UP]) and self.on_ground:
            self.velocity[1] = self.jump_speed
            self.on_ground = False

        # Gravidade como vetor.
        gravity = np.array([0.0, GRAVITY])
        self.velocity += gravity * dt

        # Atualização de posição:
        # p(t + dt) = p(t) + v * dt
        self.pos += self.velocity * dt

        self._resolve_collisions()

        # Limites horizontais do mundo.
        self.pos[0] = max(0, min(self.pos[0], WORLD_WIDTH - self.width))

        # Queda fora do mundo.
        if self.pos[1] > WORLD_HEIGHT:
            self.reset()

    def _resolve_collisions(self):
        rect = self.rect
        self.on_ground = False

        # Colisão vertical simplificada.
        for platform in platforms:
            if rect.colliderect(platform):
                previous_bottom = self.pos[1] + self.height - self.velocity[1] / FPS

                # Pousando sobre a plataforma.
                if self.velocity[1] >= 0 and previous_bottom <= platform.top + 8:
                    self.pos[1] = platform.top - self.height
                    self.velocity[1] = 0
                    self.on_ground = True
                    rect = self.rect

                # Batendo na parte inferior.
                elif self.velocity[1] < 0 and self.pos[1] >= platform.bottom - 5:
                    self.pos[1] = platform.bottom
                    self.velocity[1] = 0
                    rect = self.rect


# ------------------------------------------------------------
# [MELHORIA] Classe Projétil
# ------------------------------------------------------------

PROJECTILE_COLOR = (255, 140, 60)


class Projectile:
    RADIUS = 6
    MAX_BOUNCES = 3
    LIFETIME = 4.0

    def __init__(self, pos, velocity):
        self.pos = np.array(pos, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.bounces = 0
        self.age = 0.0
        self.alive = True

    @property
    def rect(self):
        r = self.RADIUS
        return pygame.Rect(int(self.pos[0] - r), int(self.pos[1] - r), 2 * r, 2 * r)

    def update(self, dt):
        previous_pos = self.pos.copy()

        # Movimento retilíneo uniforme: p(t + dt) = p(t) + v * dt
        self.pos += self.velocity * dt
        self.age += dt

        # Colisão com plataformas: reflexão pelo produto interno.
        for platform in platforms:
            if self.rect.colliderect(platform):
                n = collision_normal(previous_pos, platform)
                self.velocity = reflect(self.velocity, n)
                self.pos = previous_pos
                self.bounces += 1
                break

        # Bordas do mundo também refletem.
        if self.pos[0] < 0 or self.pos[0] > WORLD_WIDTH:
            self.velocity = reflect(self.velocity, E1)
            self.pos = previous_pos
            self.bounces += 1
        if self.pos[1] < 0:
            self.velocity = reflect(self.velocity, E2)
            self.pos = previous_pos
            self.bounces += 1

        if (self.bounces > self.MAX_BOUNCES
                or self.age > self.LIFETIME
                or self.pos[1] > WORLD_HEIGHT):
            self.alive = False


# ------------------------------------------------------------
# Classe Inimigo
# ------------------------------------------------------------

class Enemy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pos = np.array([560.0, 455.0], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.max_speed = 115.0
        self.vision_distance = 420.0

        # Campo de visão de 80 graus.
        self.fov_half_angle = math.radians(40.0)
        self.fov_threshold = math.cos(self.fov_half_angle)

        # Direção inicial: o jogador começa à esquerda do inimigo,
        # então o inimigo começa olhando para a esquerda.
        self.forward = np.array([-1.0, 0.0], dtype=float)

        # Ângulo usado na transformação matricial.
        self.angle = math.pi

        # [MELHORIA] Tiro e varredura do olhar.
        self.shoot_cooldown = 0.0
        self.shoot_interval = 1.6
        self.projectile_speed = 320.0
        self.scan_time = 0.0
        self.scan_amplitude = math.radians(25.0)

    @property
    def rect(self):
        return pygame.Rect(
            int(self.pos[0] - 20),
            int(self.pos[1] - 20),
            40,
            40
        )

    def update(self, player, dt, projectiles):
        self.shoot_cooldown = max(0.0, self.shoot_cooldown - dt)

        # Vetor do inimigo até o jogador.
        direction = player.center - self.pos

        distance = np.linalg.norm(direction)

        if distance > 0:
            direction_unit = direction / distance
        else:
            direction_unit = np.zeros(2)

        # Produto interno entre direção para frente
        # e direção do alvo.
        dot = float(np.dot(self.forward, direction_unit))

        # Decide se o jogador está dentro do campo de visão.
        can_see_player = (
            distance <= self.vision_distance
            and dot >= self.fov_threshold
        )

        if can_see_player:
            # O inimigo persegue o jogador.
            self.velocity = self.max_speed * direction_unit

            # O ângulo acompanha a direção.
            self.angle = math.atan2(
                direction_unit[1],
                direction_unit[0]
            )

            # Atualiza o vetor forward usando rotação.
            self.forward = rotation_matrix(self.angle) @ E1

            # [MELHORIA] Atira na direção do jogador.
            # velocidade do projétil = escalar * vetor unitário
            if self.shoot_cooldown == 0.0:
                spawn = self.pos + 30.0 * self.forward
                velocity = self.projectile_speed * direction_unit
                projectiles.append(Projectile(spawn, velocity))
                self.shoot_cooldown = self.shoot_interval
        else:
            # Patrulha simples para a esquerda.
            # O vetor forward precisa corresponder ao sentido da patrulha.
            self.velocity = np.array([-50.0, 0.0])

            # [MELHORIA] Varredura: o olhar oscila em torno de -E1.
            # theta(t) = pi + A sin(t), e forward = R(theta) E1.
            self.scan_time += dt
            offset = self.scan_amplitude * math.sin(1.5 * self.scan_time)
            self.angle = math.pi + offset
            self.forward = rotation_matrix(self.angle) @ E1

        self.pos += self.velocity * dt

        # Mantém o inimigo em uma região jogável.
        self.pos[0] = max(120, min(self.pos[0], 900))

        # Se encostar no jogador, o jogo reinicia.
        if self.rect.colliderect(player.rect):
            player.reset()
            self.reset()

        return distance, dot, can_see_player


# ------------------------------------------------------------
# Desenho do inimigo usando transformação matricial
# ------------------------------------------------------------

def draw_enemy(surface, enemy, camera_x):
    # Triângulo no sistema de coordenadas local.
    local_points = [
        (28, 0),
        (-18, -16),
        (-18, 16)
    ]

    # Transformação:
    # p_world = R(theta) S p_local + position
    world_points = transform_points(
        local_points,
        enemy.pos,
        enemy.angle,
        scale=1.0
    )

    screen_points = [
        (int(x - camera_x), int(y))
        for x, y in world_points
    ]

    pygame.draw.polygon(surface, ENEMY_COLOR, screen_points)


def draw_player(surface, player, camera_x):
    rect = player.rect.copy()
    rect.x -= int(camera_x)
    pygame.draw.rect(surface, PLAYER_COLOR, rect, border_radius=6)


def draw_platforms(surface, camera_x):
    for platform in platforms:
        rect = platform.copy()
        rect.x -= int(camera_x)
        pygame.draw.rect(surface, PLATFORM_COLOR, rect, border_radius=4)


def draw_goal(surface, camera_x):
    rect = goal_rect.copy()
    rect.x -= int(camera_x)

    pygame.draw.rect(surface, GOAL_COLOR, rect)

    # Bandeira simples.
    pygame.draw.line(
        surface,
        GOAL_COLOR,
        (rect.left, rect.top),
        (rect.left, rect.top - 60),
        4
    )

    pygame.draw.polygon(
        surface,
        GOAL_COLOR,
        [
            (rect.left + 2, rect.top - 58),
            (rect.left + 35, rect.top - 48),
            (rect.left + 2, rect.top - 38),
        ]
    )


def draw_vision(surface, enemy, camera_x, distance, dot, can_see):
    # Apenas uma visualização aproximada do campo de visão.
    # O cálculo matemático real está no método update().
    # [MELHORIA] O raio desenhado agora é o alcance real da detecção.
    radius = enemy.vision_distance

    center = (
        int(enemy.pos[0] - camera_x),
        int(enemy.pos[1])
    )

    # [MELHORIA] Cone de visão real.
    # As bordas do cone são o vetor forward girado por +-40 graus:
    #   borda = R(+-fov_half_angle) @ forward
    # O arco é feito com vários vetores R(alpha) @ forward.
    cone = [center]
    steps = 16
    for i in range(steps + 1):
        alpha = -enemy.fov_half_angle + 2 * enemy.fov_half_angle * i / steps
        edge = rotation_matrix(alpha) @ enemy.forward * radius
        cone.append((int(center[0] + edge[0]), int(center[1] + edge[1])))

    cone_layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    fill = (235, 80, 90, 70) if can_see else (240, 190, 70, 45)
    pygame.draw.polygon(cone_layer, fill, cone)
    surface.blit(cone_layer, (0, 0))
    pygame.draw.polygon(surface, VISION_COLOR, cone, 1)

    # Direção do inimigo.
    direction = enemy.forward * radius
    end = (
        int(center[0] + direction[0]),
        int(center[1] + direction[1])
    )

    pygame.draw.line(
        surface,
        VISION_COLOR,
        center,
        end,
        2
    )


def draw_projectiles(surface, projectiles, camera_x):
    for projectile in projectiles:
        center = (int(projectile.pos[0] - camera_x), int(projectile.pos[1]))
        pygame.draw.circle(surface, PROJECTILE_COLOR, center, Projectile.RADIUS)


def draw_hud(surface, player, enemy, distance, dot, can_see, projectiles):
    title = title_font.render(
        "Platform Hunter - Álgebra Linear",
        True,
        TEXT_COLOR
    )
    surface.blit(title, (18, 16))

    lines = [
        f"Player position = ({player.pos[0]:.1f}, {player.pos[1]:.1f})",
        f"Player velocity = ({player.velocity[0]:.1f}, {player.velocity[1]:.1f})",
        f"Enemy position  = ({enemy.pos[0]:.1f}, {enemy.pos[1]:.1f})",
        f"distance = {distance:.2f}",
        f"forward . direction_unit = {dot:.3f}",
        # [MELHORIA] Ângulo pelo produto interno: theta = arccos(dot).
        f"angulo forward-jogador = {math.degrees(math.acos(max(-1.0, min(1.0, dot)))):.1f} graus",
        f"field of view = {'SIM' if can_see else 'NAO'}",
        f"projeteis ativos = {len(projectiles)}",
        "A/D ou setas: mover   |   W/Espaco: pular   |   R: reiniciar",
    ]

    y = 58
    for line in lines:
        text = small_font.render(line, True, TEXT_COLOR)
        surface.blit(text, (18, y))
        y += 23


def draw_background(surface, camera_x):
    surface.fill(BG)

    # Grade para reforçar a interpretação das posições como vetores.
    grid_size = 50
    start_x = -(int(camera_x) % grid_size)

    for x in range(start_x, WIDTH, grid_size):
        pygame.draw.line(surface, (38, 44, 58), (x, 0), (x, HEIGHT), 1)

    for y in range(0, HEIGHT, grid_size):
        pygame.draw.line(surface, (38, 44, 58), (0, y), (WIDTH, y), 1)


# ------------------------------------------------------------
# Loop principal
# ------------------------------------------------------------

def main():
    player = Player()
    enemy = Enemy()
    projectiles = []

    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    player.reset()
                    enemy.reset()
                    projectiles.clear()

        keys = pygame.key.get_pressed()

        player.update(dt, keys)
        distance, dot, can_see = enemy.update(player, dt, projectiles)

        # [MELHORIA] Atualiza projéteis e verifica acerto no jogador.
        for projectile in projectiles:
            projectile.update(dt)
            if projectile.rect.colliderect(player.rect):
                player.reset()
                enemy.reset()
                projectiles.clear()
                break
        projectiles[:] = [p for p in projectiles if p.alive]

        # Vitória.
        if player.rect.colliderect(goal_rect):
            player.reset()
            enemy.reset()
            projectiles.clear()

        # Câmera acompanha o jogador.
        camera_x = player.pos[0] - WIDTH * 0.35
        camera_x = max(0, min(camera_x, WORLD_WIDTH - WIDTH))

        # Desenho.
        draw_background(screen, camera_x)
        draw_platforms(screen, camera_x)
        draw_goal(screen, camera_x)
        draw_vision(screen, enemy, camera_x, distance, dot, can_see)
        draw_enemy(screen, enemy, camera_x)
        draw_projectiles(screen, projectiles, camera_x)
        draw_player(screen, player, camera_x)
        draw_hud(screen, player, enemy, distance, dot, can_see, projectiles)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
