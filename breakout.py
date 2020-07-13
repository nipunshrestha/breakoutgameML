import os
import pygame
import neat
import time

os.environ['SDL_VIDEO_WINDOW_POS'] = "200,30"

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
GRAY = (100, 100, 100)


class Brick(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((50, 20))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y


class Paddle(pygame.sprite.Sprite):

    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.hit = False
        self.image = pygame.Surface((80, 10))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def move_left(self):
        self.rect.x -= 20

    def move_right(self):
        self.rect.x += 20

    def hitball(self):
        self.hit = True


class Ball(pygame.sprite.Sprite):
    BALL_SPEED = 5

    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.direction_x = 1
        self.direciton_y = 1
        self.image = pygame.Surface((10, 10))
        self.image.fill(WHITE)
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

    for i in range(0, 10):
        for j in range(0, 3):
            brick = Brick(70 + i * 70, 20 + j * 50)
            brick_list.add(brick)
            all_sprites.add(brick)

    score = 0
    score_surface = None

    done = False

    while not done and  ball.rect.y < 600:

        screen.fill(GRAY)
        clock.tick(60)

        for x, paddle in enumerate(paddles):
            ge[x].fitness += 0.1
            output = nets[x].activate((paddle.rect.x, abs(paddle.rect.x - ball.rect.x), abs(paddle.rect.y - ball.rect.y)))
            if output[0] > 0.5:
                if ball.direciton_y > 0:
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
            if pygame.sprite.collide_mask(ball, paddles[x]):
                ge[x].fitness += 5
                ball.flip_direction_y()
                if not paddle.hit:
                    paddles[x].hit = True

        for x, paddle in enumerate(paddles):
            if not paddles[x].hit:
                nets.pop(x)
                ge.pop(x)
                paddles.pop(x)

        collided_bricks = pygame.sprite.groupcollide(
            brick_list, ball_list, True, False, pygame.sprite.collide_mask)

        if collided_bricks:
            ball.flip_direction_y()
            score += len(collided_bricks)
            for g in ge:
                g.fitness += len(collided_bricks)

        if score_surface is None or collided_bricks:
            score_surface = game_font.render(
                'Score: %d' % (score), False, GREEN)

        ball.move()
        if ball.leaves_screen_bottom():
            for x, paddle in enumerate(paddles):
                ge[x].fitness -= len(brick_list)
                nets.pop(x)
                ge.pop(x)
                paddles.pop(x)

        for x,paddle in enumerate(paddles):
            if paddle.rect.x + 70 >= 800 or paddle.rect.x < 0:
                nets.pop(x)
                ge.pop(x)
                paddles.pop(x)

        all_sprites.update()
        all_sprites.draw(screen)

        screen.blit(score_surface, (50, 240))
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


