if __name__ == '__main__':
    import retro
    import time
    import sys
    import ctypes
    import numpy as np
    from retro.rendering import SimpleImageViewer
    import pyglet
    from pyglet import gl
    from pyglet.window import key as keycodes

    ROM_PATH = './rom.gba'
    STATE_FILE = 'game_state.sav'

    start_time = time.time()
    em = retro.RetroEmulator(ROM_PATH)
    # viewer = SimpleImageViewer()
    inttype=retro.data.Integrations.STABLE

    # button meanings
    system = retro.get_romfile_system(ROM_PATH)
    core = retro.get_system_info(system)
    buttons = core["buttons"]
    print(buttons)

    # initialize window
    def _on_close():
        em.stop()
        window.close()
        sys.exit(0)

    em.step()
    img = em.get_screen()
    # get img width and height
    image_width, image_height = img.shape[1], img.shape[0]
    window = pyglet.window.Window(width=image_width, height=image_height)
    _key_handler = pyglet.window.key.KeyStateHandler()
    window.push_handlers(_key_handler)
    window.on_close = _on_close

    gl.glEnable(gl.GL_TEXTURE_2D)
    _texture_id = gl.GLuint(0)
    gl.glGenTextures(1, ctypes.byref(_texture_id))
    gl.glBindTexture(gl.GL_TEXTURE_2D, _texture_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        gl.GL_RGBA8,
        image_width,
        image_height,
        0,
        gl.GL_RGB,
        gl.GL_UNSIGNED_BYTE,
        None,
    )

    def _draw(img):
        gl.glBindTexture(gl.GL_TEXTURE_2D, _texture_id)
        video_buffer = ctypes.cast(
            img.tobytes(),
            ctypes.POINTER(ctypes.c_short),
        )
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            img.shape[1],
            img.shape[0],
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            video_buffer,
        )

        x = 0
        y = 0
        w = window.width
        h = window.height

        pyglet.graphics.draw(
            4,
            pyglet.gl.GL_QUADS,
            ("v2f", [x, y, x + w, y, x + w, y + h, x, y + h]),
            ("t2f", [0, 1, 1, 1, 1, 0, 0, 0]),
        )
    
    while True:
        em.step()
        img = em.get_screen()
        # viewer.imshow(img)
        window.switch_to()
        window.dispatch_events()
        _draw(img)
        window.flip()
        if time.time() - start_time >= 60 * 5:
            with open(STATE_FILE, 'wb') as fp:
                em.save_state(fp)
                em.stop()
                exit()