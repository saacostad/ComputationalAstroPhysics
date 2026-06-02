import sqlite3 as sql 

# Nos conectamos a la base de datos 
with sql.connect("shows.db") as conn:
    
    # Creamos el agente
    cursor = conn.cursor()

    """
    PRIMER EJECICIO: cuántos con calificación perfecta 

    Las calificaciones están en la tabla `ratings.rating`, entonces toca contar cuántas tienen rate de 10.0
    """
    
    # COUNT sirve para contar el total en vez de guardar la información
    cursor.execute("""
    SELECT COUNT (*) FROM ratings WHERE rating = ? 
    """, (10.0, ))
    
    count = cursor.fetchone()[0]
    print(f"Primer ejecicio | cuántos shows con calificación perfecta: {count}")


    """ 
    SEGUNDO EJERCICIO: número de episodios de Black Mirror

    En este caso, tanto el nombre de la serie como el número de episodios están en la tabla `shows`
    """

    cursor.execute("""
    SELECT episodes FROM shows WHERE title LIKE ?
    """, ("Black Mirror", ))

    count = cursor.fetchone()[0]
    print(f"Segundo ejecicio | # episodios de Black Mirror: {count}")


    """
    TERCER EJERCICIO: Series existentes en género "ciencia ficción"

    En este caso, tenemos que contar cuantas series están en ciencia ficción. Para ello, vamos a contar cuántas 
    filas aparecen cuando seleccionamos el género `Sci-Fi`.
    """
    
    cursor.execute("""
    SELECT COUNT (*) FROM genres WHERE genre LIKE ?
    """, ("Sci-Fi", ))

    count = cursor.fetchone()[0]
    print(f"Tercer ejecicio | # series Sci-Fi: {count}")


    """
    CUARTO EJERCICIO: serie mejor valorada de terror
    En este caso, tenemos que combinar tres tablas: `shows.title`, `genres.genre` y `ratings.rating`. 
    Obtener Sólo las filas con genre Terror y obtener el title con el max(rating).
    Tenemos que unir todas estas tablas según el ID 
    """

    cursor.execute("""
    SELECT shows.title, ratings.rating 
    FROM shows
    JOIN ratings ON shows.id = ratings.show_id
    JOIN genres ON shows.id = genres.show_id
    WHERE genres.genre LIKE ?
    AND ratings.rating = (
        SELECT MAX(ratings.rating)
        FROM ratings
        JOIN genres ON genres.show_id = ratings.show_id
        WHERE genres.genre LIKE ?
    )
    """, ("Thriller", "Thriller",))

    query = cursor.fetchall()[0]

    print(f"Cuarto ejecicio | Mejor serie de Thriller: {query[0]} con rating de {query[1]}")


    """
    QUINTO EJERCICIO: cuántas series de animación hay]
    Este es más sencillo, no es más que contar la cantidad de series de animación 
    """

    cursor.execute("""
    SELECT COUNT (*) FROM genres WHERE genre LIKE ?
    """, ("Animation", ))
    
    query = cursor.fetchone()[0]

    print(f"Quinto ejecicio | Cantidad de series de animación: {query}")


    """
    SEXTO EJERCICIO: las 10 peores series entre 2005 y 2010
    Este es más complicato. Hay que combinar las tablas de `shows` y `ratings`, filtrar
    las series entre 2005 y 2010 y organizarlas según el rating
    """

    cursor.execute("""
    SELECT shows.title, shows.year, ratings.rating
    FROM shows
    JOIN ratings ON shows.id = ratings.show_id
    WHERE shows.year BETWEEN ? AND ?
    ORDER BY ratings.rating ASC
    LIMIT 10
    """, (2005, 2010, ))
    
    query = cursor.fetchall()

    print(f"Sexto ejercicio | 10 peores series entre 2005 y 2010:")

    for row in query:
        print(f"{row[0]} del año {row[1]} con rating de {row[2]}")
