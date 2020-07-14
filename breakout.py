import os
import pygame
import neat
import math

os.environ['SDL_VIDEO_WINDOW_POS'] = "200,30"

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
GRAY = (100, 100, 100)
gen = 0


class Util():
    def intersects(self, a, b, c, min, max):
        disc = b ** 2 - 4 * a * c
        if disc < 0:
            return False
        s1 = (-b + math.sqrt(disc)) / (2 * a)
        s2 = (-b - math.sqrt(disc)) / (2 * a)
        return (min <= s1 <= max) or (min <= s2 <= max)


class Brick(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((60, 10))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y


class Paddle(pygame.sprite.Sprite):
    WIDTH = 40
    HEIGHT = 5

    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.hit = False
        self.image = pygame.Surface((80, 10))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.util = Util()

    def move_left(self):
        self.rect.x -= 8

    def move_right(self):
        self.rect.x += 8

    def collides_ball(self, ball):
        intersects_x = False
        intersects_y = False

        intersects_x = intersects_x or self.util.intersects(
            1, -2 * ball.rect.x, (ball.rect.y - self.rect.y) ** 2 - 10 ** 2 + ball.rect.x ** 2, self.rect.x,
            self.rect.x + self.WIDTH)
        intersects_y = intersects_y or self.util.intersects(
            1, -2 * ball.rect.y, (ball.rect.x - self.rect.x) ** 2 - 10 ** 2 + ball.rect.y ** 2, self.rect.y,
            self.rect.y + self.HEIGHT)
        intersects_y = intersects_y or self.util.intersects(
            1, -2 * ball.rect.y, (ball.rect.x - self.rect.x - self.WIDTH) ** 2 - 10 ** 2 + ball.rect.y ** 2, self.rect.y,
            self.rect.y + self.HEIGHT)

        return intersects_x, intersects_y


class Ball(pygame.sprite.Sprite):
    BALL_SPEED = 5

    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.direction_x = 1
        self.direciton_y = 1
        self.image = pygame.Surface([10, 10])
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def flip_direction_x(self):
        self.direction_x *= -1

    def flip_direction_y(self):
        self.direciton_y *= -1

    def leaves_screen_bottom(self):
        if self.rect.x < 0 or self.rect.x > SCREEN_WIDTH:
            self.flip_direction_x()
        if self.rect.y < 0:
            self.flip_direction_y()

        return self.rect.y > SCREEN_HEIGHT

    def move(self):
        self.rect.x += self.BALL_SPEED * self.direction_x
        self.rect.y += self.BALL_SPEED * self.direciton_y


def eval_genomes(genomes, config):
    pygame.init()
    pygame.font.init()
    game_font = pygame.font.SysFont('', 30)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Breakout Game')
    clock = pygame.time.Clock()
    ge = []
    nets = []
    paddles = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        paddles.append(Paddle(100, 550))
        g.fitness = 0
        ge.append(g)

    ball = Ball(250, 250)

    brick_list = pygame.sprite.Group()
    paddle_list = pygame.sprite.Group()
    ball_list = pygame.sprite.Group()
    all_sprites = pygame.sprite.Group()

    for paddle in paddles:
        paddle_list.add(paddle)
        all_sprites.add(paddle)

    ball_list.add(ball)
    all_sprites.add(ball)

    for i in range(0, 11):
        for j in range(0, 7):
            brick = Brick(20 + i * 70, 15 + j * 20)
            brick_list.add(brick)
            all_sprites.add(brick)

    global gen
    gen += 1
    score = 0
    score_surface = None
    alive_surface = None
    gen_surface = None

    done = False

    while not done and len(paddles)>0:

        screen.fill(GRAY)
        clock.tick(60)

        for x, paddle in enumerate(paddles):
            ge[x].fitness += 0.1
            output = nets[x].activate((paddle.rect.x, abs(paddle.rect.x - ball.rect.x), abs(paddle.rect.y - ball.rect.y)))
            if output[0] > 0.5:
                if ball.direciton_y > 0:
                    if ball.rect.y + 10 == paddle.rect.y:
                        continue
                    if ball.direction_x > 0:
                        paddle.move_right()
                    else:
                        paddle.move_left()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                pygame.quit()
                quit()
                break

        for x, paddle in enumerate(paddles):
            paddle_ball_intersects_x, paddle_ball_intersects_y = paddle.collides_ball(ball)
            if paddle_ball_intersects_y:
                ge[x].fitness += 5
                ball.flip_direction_y()

            if ball.rect.y + 5 >= paddle.rect.y and not paddle_ball_intersects_y:
                ge[x].fitness -= 5
                nets.pop(x)
                ge.pop(x)
                paddles.pop(x)
                paddle_list.remove(paddle)
                all_sprites.remove(paddle)

        collided_bricks = pygame.sprite.groupcollide(
            brick_list, ball_list, True, False, pygame.sprite.collide_mask)

        if collided_bricks:
            ball.flip_direction_y()
            score += len(collided_bricks)
            for g in ge:
                g.fitness += 1

        if score_surface is None or collided_bricks:
            score_surface = game_font.render('Score: %d' % score, False, GREEN)

        if alive_surface is None or len(paddles) > 0:
            alive_surface=game_font.render('Alive: %d' % len(paddles), False, GREEN)

        if gen_surface is None or gen >= 0:
            gen_surface = game_font.render('Generation: %d' % gen, False, GREEN)

        ball.move()
        if ball.leaves_screen_bottom():
            for x, paddle in enumerate(paddles):
                ge[x].fitness -= len(brick_list)
                nets.pop(x)
                ge.pop(x)
                paddles.pop(x)
                paddle_list.remove(paddle)
                all_sprites.remove(paddle)

        for x, paddle in enumerate(paddles):
            if paddle.rect.x + 80 > 810 or paddle.rect.x < 0:
                ge[x].fitness -= 5
                nets.pop(x)
                ge.pop(x)
                paddles.pop(x)
                all_sprites.remove(paddle)
                paddle_list.remove(paddle)

        all_sprites.update()
        all_sprites.draw(screen)

        screen.blit(score_surface, (50, 240))
        screen.blit(alive_surface, (50, 270))
        screen.blit(gen_surface, (50, 300))
        pygame.display.update()


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 50)
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "break_config.txt")
    run(config_path)


