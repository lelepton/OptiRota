def _normalize_tag(tag: str) -> str:
    '''Normaliza a tag de highway para um dos valores-base esperados
    ("primary", "secondary", "tertiary", "residential"). Qualquer outra
    tag cai em "residential" por padrão.

    Parâmetros
    ----------
    tag : str (tag original do CSV)

    Retorno
    ---------
    str : tag normalizada
    '''
    
    t = tag.strip().lower()
    return t if t in {"primary", "secondary", "tertiary", "residential"} else "residential"
