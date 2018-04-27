from views import index
from aiohttp import web

def setup_routes(app):
	app.router.add_get('/', index)