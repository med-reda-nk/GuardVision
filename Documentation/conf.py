# conf.py

# -- General configuration ------------------------------------------------

extensions = [
    "sphinx_rtd_theme",  # Ajout du thème ReadTheDocs
]

# The master toctree document.
master_doc = 'index'

# Informations générales
project = 'GuardVision'
copyright = '2023, Mohamed Reda Nkira'
author = 'Mohamed Reda Nkira - Zouga Mouhcine'

version = '0.1'
release = '0.1.0'

language = 'fr'  # Spécifie le français pour la génération de contenu

# -- Options pour le rendu HTML -------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 3,
    'style_external_links': True,
}

html_static_path = ['_static']  # Assure-toi que ce dossier existe ou enlève cette ligne
htmlhelp_basename = 'GuardVisionDoc'

# -- Options pour le rendu LaTeX ------------------------------------------

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'preamble': '',
}

latex_documents = [
    (master_doc, 'GuardVision.tex', 'Documentation GuardVision',
     author, 'manual'),
]

# -- Options pour le rendu man --------------------------------------------

man_pages = [
    (master_doc, 'guardvision', 'Documentation GuardVision',
     [author], 1)
]

# -- Options pour le rendu Texinfo ----------------------------------------

texinfo_documents = [
    (master_doc, 'GuardVision', 'Documentation GuardVision',
     author, 'GuardVision', 'Système de surveillance intelligent.',
     'Miscellaneous'),
]
