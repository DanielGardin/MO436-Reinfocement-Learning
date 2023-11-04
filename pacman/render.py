import time

SLEEP_TIME = 0

class RenderMode:
    @staticmethod
    def render(state):
        return NotImplemented

    @staticmethod
    def close():
        return NotImplemented

class AnsiRender:
    @staticmethod
    def render(state):
        print(state)
    
    @staticmethod
    def close():
        pass