# All models need to be imported in the correct order to resolve relationships
# Import base class first
from .base_models import *

# Import models in dependency order (least dependent first)
from .employee import *
from .guest import *
from .customer import *
from .document import *
from .feedback import *
from .product import *

# Ensure all relationships are properly configured
from sqlalchemy.orm import configure_mappers
configure_mappers()
