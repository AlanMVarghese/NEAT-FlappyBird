import pygame
import neat
import os
import random
import sys

pygame.init()
WIN_WIDTH = 500
WIN_HEIGHT = 700
GEN = -1
ALIVE = 0
SCREEN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
FONT = pygame.font.SysFont("comicsans", 25)
FONTSM = pygame.font.SysFont("comicsans", 20)
BIRD = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
        pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
        pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]
BOTTOM_PIPE = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
TOP_PIPE = pygame.transform.flip(BOTTOM_PIPE, False, True)
BASE = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

class Bird:
    X_POS = 80
    Y_POS = 310
    VEL = 8.5
    GRAVITY = 1
    JUMP_VEL = -10.5
    def __init__(self, img=BIRD[0]):
        self.image = img
        self.vel = 0
        self.bird_jump = False
        self.rect = pygame.Rect(self.X_POS, self.Y_POS, img.get_width(), img.get_height())
        self.imgcount = 0
    def update(self):
        self.rect.y += self.vel
        if self.rect.y < 0:
            self.rect.y = 0
        if self.rect.y > WIN_HEIGHT - self.rect.height:
            self.rect.y = WIN_HEIGHT - self.rect.height
        if self.vel < 15:
            self.vel += self.GRAVITY
        if self.bird_jump:
            self.jump()
        else:
            self.normal()
        if self.imgcount >= 9:
            self.imgcount = 0
    def jump(self):
        self.vel = self.JUMP_VEL
        self.bird_jump = False
    def normal(self):
        self.image = BIRD[self.imgcount // 3]
        self.rect.x = self.X_POS
        self.imgcount += 1
    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.rect.x, self.rect.y))

class Pipe:
    GAP = 170
    def __init__(self, x=500, imgbottom=BOTTOM_PIPE, imgtop=TOP_PIPE):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.toppipe1 = imgtop
        self.bottompipe1 = imgbottom
        self.passed = False
        self.set_height1()
    def set_height1(self):
        self.height = random.randint(40, 300)
        self.top = self.height - self.toppipe1.get_height()
        self.bottom = self.height + self.GAP
    def update(self):
        self.x -= 5
        if self.x < -BOTTOM_PIPE.get_width():
            self.x = 500
            self.set_height1()
            self.passed = False
    def draw1(self, SCREEN):
        SCREEN.blit(self.toppipe1, (self.x, self.top))
        SCREEN.blit(self.bottompipe1, (self.x, self.bottom))
    def get_rects(self):
        top_rect = self.toppipe1.get_rect(topleft=(self.x, self.top))
        bottom_rect = self.bottompipe1.get_rect(topleft=(self.x, self.bottom))
        return top_rect, bottom_rect

def eval_genomes(genomes, config):
    global x_pos_base, y_pos_base, x_pos_bg, y_pos_bg,GEN,ALIVE
    GEN += 1
    nets = []
    ge = []
    birds = []
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird())
        g.fitness = 0
        ge.append(g)
    x_pos_base = 0
    y_pos_base = 580
    x_pos_bg = 0
    y_pos_bg = -200
    clock = pygame.time.Clock()
    pipes = [Pipe(), Pipe(x=800)]
    score = 0
    def base(): 
        global x_pos_base, y_pos_base
        image_width = BASE.get_width()
        SCREEN.blit(BASE, (x_pos_base, y_pos_base))
        SCREEN.blit(BASE, (image_width + x_pos_base, y_pos_base))
        if x_pos_base <= -image_width:
            x_pos_base = 0
        x_pos_base -= 5
    def bg():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            x_pos_bg = 0
        x_pos_bg -= 1
    def draw_score(score):
        score_text = FONT.render("Score: " + str(score), 1, (255, 255, 255))
        SCREEN.blit(score_text, (WIN_WIDTH - score_text.get_width() - 10, 10))
        gen_text = FONT.render("Gen " + str(GEN), 1, (255, 255, 255))
        SCREEN.blit(gen_text, (10, 10))
        alive_text = FONTSM.render("Alive " + str(len(birds)), 1, (255, 255, 255))
        SCREEN.blit(alive_text, (10, 40))
    run = True
    pipe_ind = 0
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        if len(birds)>0:
            if 80 > pipes[pipe_ind].x + TOP_PIPE.get_width():
                if pipe_ind == 0:
                    pipe_ind = 1
                else:
                    pipe_ind = 0
        else:
            run = False
            break
        for x, bird in enumerate(birds):
            ge[x].fitness += 0.1
            output = nets[x].activate((bird.rect.y, abs(bird.rect.y - pipes[pipe_ind].bottom), abs(bird.rect.y - pipes[pipe_ind].top)))
            if output[0] > 0.5:
                bird.jump() 
        for pipe in pipes:
            for x,bird in enumerate(birds): 
                top_rect, bottom_rect = pipe.get_rects()
                if bird.rect.colliderect(top_rect) or bird.rect.colliderect(bottom_rect):
                    ge[x].fitness -= 1
                    nets.pop(x)
                    ge.pop(x)  
                    birds.pop(x)
                if not pipe.passed and bird.rect.x > pipe.x+pipe.toppipe1.get_width():
                    pipe.passed = True
                    score += 1
                    for g in ge:
                        g.fitness += 5
                if bird.rect.y <= 0 or bird.rect.y >= WIN_HEIGHT + bird.rect.height - BASE.get_height():
                      ge[x].fitness -= 1
                      nets.pop(x)
                      ge.pop(x)  
                      birds.pop(x)
        bg()
        for pipe in pipes:    
            pipe.update() 
            pipe.draw1(SCREEN)
        base()
        for bird in birds:
            bird.update()
            bird.draw(SCREEN)
        draw_score(score)
        clock.tick(30)
        pygame.display.update()
def run(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.run(eval_genomes, 50)
if __name__ == '__main__':
    local_directory = os.path.dirname(__file__)
    config_path = os.path.join(local_directory, "neatconfig.txt")
    run(config_path)