#!/usr/bin/env python

# mandelbulb slices rendered in gpu shader, heavily based on vispy gallery 30
# example

from vispy import app, gloo

vertex = """
attribute vec2 position;
void main()
{
    gl_Position = vec4(position, 0, 1.0);
}
"""

fragment = """
uniform vec2 resolution;
uniform vec3 center;
uniform float scale;
vec3 hot(float t)
{
    return vec3(smoothstep(0.00,0.33,t),
                smoothstep(0.33,0.66,t),
                smoothstep(0.66,1.00,t));
}
void main()
{

    const int n_iter = 300;
    const float log_2 = 0.6931471805599453;
    vec3 c;

    float ct, st, cp, sp, rn;
    float r, theta, phi, xsqr, ysqr;
    int i;
    int n;
    float x, y, z, d;

    // Recover coordinates from pixel coordinates
    c.x = (gl_FragCoord.x / resolution.x - 0.5) * scale + center.x;
    c.y = (gl_FragCoord.y / resolution.y - 0.5) * scale + center.y;
    c.z = center.z;

    x = c.x;
    y = c.y;
    z = c.z;
    n = 8;

    // vec3 z = c;
    for(i = 0; i < n_iter; ++i)
    {
        xsqr = x * x;
        ysqr = y * y;
        r = sqrt(xsqr + ysqr + z*z);
        theta = atan(sqrt(xsqr + ysqr), z);
        phi = atan(y, x);

        ct = cos(theta * n);
        st = sin(theta * n);
        cp = cos(phi * n);
        sp = sin(phi * n);
        rn = pow(r, n);

        x = rn * st * cp + c.x;
        y = rn * st * sp + c.y;
        z = rn * ct + c.z;

        d = x*x + y*y + z*z;
        if (d > 2.0) break;
        // z = vec3(x, y, z);
    }
    if ( i < n_iter ) {
        float nu = log(log(sqrt(d))/log_2)/log_2;
        float index = float(i) + 1.0 - nu;
        float v = pow(index/float(n_iter),0.5);
        gl_FragColor = vec4(hot(v),1.0);
    } else {
        gl_FragColor = vec4(hot(0.0),1.0);
    }
}
"""


# vispy Canvas
# -----------------------------------------------------------------------------
class Canvas(app.Canvas):

    def __init__(self, *args, **kwargs):
        app.Canvas.__init__(self, *args, **kwargs)
        self.program = gloo.Program(vertex, fragment)

        # Draw a rectangle that takes up the whole screen. All of the work is
        # done in the shader.
        self.program["position"] = [(-1, -1), (-1, 1), (1, 1),
                                    (-1, -1), (1, 1), (1, -1)]

        self.scale = self.program["scale"] = 3
        self.center = self.program["center"] = [-0.5, 0, 0]
        self.apply_zoom()

        self.bounds = [-2, 2]
        self.min_scale = 1e-20
        self.max_scale = 5

        gloo.set_clear_color(color='black')

        self._timer = app.Timer('auto', connect=self.update, start=True)

        self.show()

    def on_draw(self, event):
        self.program.draw()

    def on_resize(self, event):
        self.apply_zoom()

    def apply_zoom(self):
        width, height = self.physical_size
        gloo.set_viewport(0, 0, width, height)
        self.program['resolution'] = [width, height]

    def on_mouse_move(self, event):
        """Pan the view based on the change in mouse position."""
        if event.is_dragging and event.buttons[0] == 1:
            x0, y0 = event.last_event.pos[0], event.last_event.pos[1]
            x1, y1 = event.pos[0], event.pos[1]
            X0, Y0, Z0 = self.pixel_to_coords(float(x0), float(y0))
            X1, Y1, Z1 = self.pixel_to_coords(float(x1), float(y1))
            self.translate_center(X1 - X0, Y1 - Y0, Z1 - Z0)

    def translate_center(self, dx, dy, dz):
        """Translates the center point, and keeps it in bounds."""
        center = self.center
        center[0] -= dx
        center[1] -= dy
        center[2] -= dz
        center[0] = min(max(center[0], self.bounds[0]), self.bounds[1])
        center[1] = min(max(center[1], self.bounds[0]), self.bounds[1])
        center[2] = min(max(center[2], self.bounds[0]), self.bounds[1])
        self.program["center"] = self.center = center

    def pixel_to_coords(self, x, y):
        """Convert pixel coordinates to Mandelbrot set coordinates."""
        rx, ry = self.size
        nx = (x / rx - 0.5) * self.scale + self.center[0]
        ny = ((ry - y) / ry - 0.5) * self.scale + self.center[1]
        nz = self.center[2]
        return [nx, ny, nz]

    def on_mouse_wheel(self, event):
        """Use the mouse wheel to zoom."""
        delta = event.delta[1]
        if delta > 0:  # Zoom in
            factor = 0.9
        elif delta < 0:  # Zoom out
            factor = 1 / 0.9
        for _ in range(int(abs(delta))):
            self.zoom(factor, event.pos)

    def on_key_press(self, event):
        """Use + or - to zoom in and out.
        The mouse wheel can be used to zoom, but some people don't have mouse
        wheels :)
        """

        if event.text == '+' or event.text == '=':
            self.zoom(0.9)
        elif event.text == '-':
            self.zoom(1/0.9)
        elif event.text == 'l':
            self.translate_center(0.0, 0.0, -0.01)
        elif event.text == 'L':
            self.translate_center(0.0, 0.0, 0.01)

    def zoom(self, factor, mouse_coords=None):
        """Factors less than zero zoom in, and greater than zero zoom out.
        If mouse_coords is given, the point under the mouse stays stationary
        while zooming. mouse_coords should come from MouseEvent.pos.
        """
        if mouse_coords is not None:  # Record the position of the mouse
            x, y = float(mouse_coords[0]), float(mouse_coords[1])
            x0, y0, z0 = self.pixel_to_coords(x, y)

        self.scale *= factor
        self.scale = max(min(self.scale, self.max_scale), self.min_scale)
        self.program["scale"] = self.scale

        # Translate so the mouse point is stationary
        if mouse_coords is not None:
            x1, y1, z1 = self.pixel_to_coords(x, y)
            self.translate_center(x1 - x0, y1 - y0, z1 - z0)


if __name__ == '__main__':
    canvas = Canvas(size=(1024, 1024), keys='interactive')
    app.run()
