from aiohttp import web
import aiohttp_jinja2

async def index(request):
	return aiohttp_jinja2.render_template('main.html',
                                              request,
                                              {})