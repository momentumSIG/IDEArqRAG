"""RAG-IDEArq-eval-v3 — 30 preguntas canónicas (15 simples + 15 complejas).

Distribución por idioma (proporcional al corpus LDA, 513 docs):
  ES: 55% → 16 (8 simples + 8 complejas)
  EN: 33% → 10 (5 simples + 5 complejas)
  PT:  6% →  2 (1 simple  + 1 compleja)
  CA:  4% →  1 (1 simple)
  FR:  1% →  1 (1 compleja)

Cada item: (id, pregunta, ground_truth, fuente, tipo, idioma, n_articulos)
"""

EVAL_ITEMS = [
    # ═══════════════════════════════════════════════════════════════════
    # ESPAÑOL — Simples (8)
    # ═══════════════════════════════════════════════════════════════════
    (0,
     "¿En qué año excavaron en el yacimiento de La Bastida de Totana los hermanos Siret?",
     "En el año 1886.",
     "manual_review", "simple", "es", 1),

    (1,
     "¿Cuál es el yacimiento calcolítico más alejado de la ciudad de Jaén en la provincia de Jaén?",
     "El yacimiento de Eras del Alcázar, a aproximadamente 50 km de la ciudad de Jaén.",
     "manual_review", "simple", "es", 1),

    (2,
     "Excavaciones de urgencia de la Junta de Andalucía en la provincia de Almería publicadas en 2001 — ¿qué solares?",
     "Solar en la avenida Pablo Iglesias esquina A Rafaela Jiménez, solar en la calle La Central de Villaricos (cuevas de Almanzora) y en la calle Castillejo (Gádor, Almería).",
     "manual_review", "simple", "es", 1),

    (3,
     "¿Es correcta la datación -1377±23 para Valencina Cerro de la Cabeza Ladera Sur como la más reciente del área?",
     "No, es errónea. La datación correcta más reciente es: Valencina, Cerro de la Cabeza, Ladera Sur, 175 ± 20.",
     "manual_review", "simple", "es", 1),

    (4,
     "¿Estas dataciones son del Neolítico? (CV 33.900±60, Galicia 31.690±50, Murcia 12.030±0)",
     "No, estas dataciones corresponden al período Paleolítico.",
     "manual_review", "simple", "es", 1),

    (5,
     "¿Cuál es la fecha más antigua para un yacimiento funerario megalítico en la Península Ibérica?",
     "Trikuaizti 2 (Gipuzkoa), 12015 ± 145.",
     "manual_review", "simple", "es", 1),

    (6,
     "Datación más reciente de los yacimientos calcolíticos en el área de Valencina de la Concepción (Sevilla).",
     "Valencina, Cerro de la Cabeza, Ladera Sur, 175 ± 20.",
     "manual_review", "simple", "es", 1),

    (7,
     "Yacimiento con la data de C14 más antigua de las Islas Baleares.",
     "Cova de Moleta (Sóller), 80000 BP.",
     "manual_review", "simple", "es", 1),

    # ═══════════════════════════════════════════════════════════════════
    # ESPAÑOL — Complejas (8)
    # ═══════════════════════════════════════════════════════════════════
    (8,
     "¿Cuáles son las cronologías de las manifestaciones funerarias del Mesolítico en las distintas regiones peninsulares?",
     "Mediterráneo: cementerios ~9475-9300 cal BP (El Collado). Portugal (Muge): ~8409-8030 cal BP. Cantábrico: ~7981-6636 cal BP.",
     "manual_review", "compleja", "es", 3),

    (9,
     "Periodización del Bronce Final en el Levante de la Península Ibérica, cronología de las fases y principales ejemplos de yacimientos.",
     "Bronce tardío (c.1550/1500-1300/1250): Oropesa la Vella, Cabezo Redondo. Bronce Final I (1300-1000): Costamar, Cova d'en Pardo. Bronce Final II (1000-850): Ereta del Castellar, Mola d'Agres. Bronce Final III (850-725): La Vital, Peña Negra I. Hierro antiguo (725-550): Vinarragell, El Molón.",
     "manual_review", "compleja", "es", 5),

    (10,
     "Yacimientos Calcolíticos de la Península Ibérica en los que se han hallado objetos de marfil.",
     "Pre-campaniforme: Zambujal, Vila Nova de São Pedro, Leceia, Alcalar, Perdigões, Valencina, Los Millares. Campaniforme: Palmela, Pedra do Ouro, Verdelha dos Ruivos, Los Algarbes, Cerro de la Virgen.",
     "manual_review", "compleja", "es", 4),

    (11,
     "Cronología y distribución espacial del poblamiento neolítico en la Meseta Sur.",
     "Valle del Tajo: cuevas La Ventana, La Higuera (Sierra madrileña). La Mancha: ocupaciones en cuevas y abrigos.",
     "manual_review", "compleja", "es", 5),

    (12,
     "Yacimientos neolíticos situados a menos de 150 km de Casa Montero.",
     "24 sitios: La Atalaya (Ávila), Portillo de las Cortes (Guadalajara), El Cañaveral, Casa Montero, Cueva de la Higuera (Madrid), Cueva de la Vaquera (Segovia), La Mina, Peña de la Abuela (Soria), El Castillejo (Toledo), entre otros.",
     "manual_review", "compleja", "es", 5),

    (13,
     "Datación más antigua y más reciente de los yacimientos calcolíticos en el área de Valencina de la Concepción (Sevilla).",
     "Antigua: Valencina, Instituto de Educación Secundaria, 4800 ± 100. Reciente: Valencina, Cerro de la Cabeza, Ladera Sur, 175 ± 20.",
     "manual_review", "compleja", "es", 2),

    (14,
     "Dataciones más antiguas (i.e, más altas) de yacimientos paleolíticos para cada comunidad autónoma.",
     "Andalucía 51.914±45, Aragón 25.330±80, Cantabria 48.200±80, Castilla y León 30.300±25, Castilla-La Mancha 28.660±40, Cataluña 38.640±50, C. Madrid 30.280±28, Navarra 21.600±30, C. Valenciana 33.900±60, Extremadura 61.219±70, Galicia 31.690±50, Illes Balears 24.220±115, La Rioja 6.220±100, País Vasco 34.350±130, Asturias 16.700±30, Murcia 12.030±0.",
     "manual_review", "compleja", "es", 10),

    (15,
     "Principales yacimientos de la Segunda Edad del Hierro en la provincia de León.",
     "Valencia de Don Juan, Lancia, Regueras de Arriba, Castros del Teleno/Valdería/Bierzo, Castro de Chano, Peña del Castro (La Ercina), El Castrelín de San Juan de Paluezas, entre otros (15+ sitios).",
     "manual_review", "compleja", "es", 3),

    # ═══════════════════════════════════════════════════════════════════
    # INGLÉS — Simples (5)
    # ═══════════════════════════════════════════════════════════════════
    (16,
     "In what year did the Siret brothers excavate the La Bastida de Totana site?",
     "In 1886.",
     "manual_review", "simple", "en", 1),

    (17,
     "What is the oldest C14 date for the Balearic Islands?",
     "Cova de Moleta (Sóller), 80000 BP.",
     "manual_review", "simple", "en", 1),

    (18,
     "When was the Casa Montero flint mine active?",
     "Main episode: 5327-5215 cal BC (1σ), lasting just over a century.",
     "manual_review", "simple", "en", 1),

    (19,
     "Which site in Sevilla has both the oldest and most recent C14 dates for the Chalcolithic?",
     "Valencina de la Concepción (oldest: IES 4800±100; most recent: Cerro de la Cabeza 175±20).",
     "manual_review", "simple", "en", 1),

    (20,
     "What is the oldest Paleolithic date for Andalucía?",
     "Andalucía, 51.914 ± 45.",
     "manual_review", "simple", "en", 1),

    # ═══════════════════════════════════════════════════════════════════
    # INGLÉS — Complejas (5)
    # ═══════════════════════════════════════════════════════════════════
    (21,
     "What are the main theoretical models of Neolithic expansion in Europe?",
     "Demic diffusion (movement of Neolithic societies and agricultural practices) vs Cultural diffusion (transmission of the Neolithic package: technology, pottery, plants, domesticated animals).",
     "manual_review", "compleja", "en", 2),

    (22,
     "Main Iron Age II sites in León province.",
     "15+ sites: Valencia de Don Juan, Lancia, Regueras de Arriba, Castros del Teleno/Valdería/Bierzo, Castro de Chano, Peña del Castro, El Castrelín de San Juan de Paluezas, La Corona del Castro, La Peña del Hombre, Castro de Columbrianos, Peña Piñera.",
     "manual_review", "compleja", "en", 3),

    (23,
     "Chalcolithic sites with ivory objects in the Iberian Peninsula.",
     "Pre-Bell Beaker (12 sites): Zambujal, Vila Nova de São Pedro, Leceia, Praia das Maçãs, Palmela, Alcalar, Perdigões, Señorío de Guzmán, La Pijotilla, Valencina, Gilena, Los Millares. Bell Beaker (10 sites): Palmela, Pedra do Ouro, Verdelha dos Ruivos, VNSP, Perdigões, Valencina, Los Algarbes, Cerro de la Virgen, Camino de Yeseras, La Pijotilla.",
     "manual_review", "compleja", "en", 4),

    (24,
     "Oldest Paleolithic dates by autonomous community in the Iberian Peninsula.",
     "Andalucía 51.914±45, Aragón 25.330±80, Cantabria 48.200±80, Castilla y León 30.300±25, Castilla-La Mancha 28.660±40, Cataluña 38.640±50, C. Madrid 30.280±28, Navarra 21.600±30, C. Valenciana 33.900±60, Extremadura 61.219±70, Galicia 31.690±50, Illes Balears 24.220±115, La Rioja 6.220±100, País Vasco 34.350±130, Asturias 16.700±30, Murcia 12.030±0.",
     "manual_review", "compleja", "en", 10),

    (25,
     "Main funerary chronologies of the Mesolithic across peninsular regions.",
     "Mediterranean: El Collado 9475-9300 cal BP, Casa Corona/Cingle del Mas Nou 8007-7583 cal BP. Atlantic Portugal: Muge estuary 8409-8030 cal BP (Cabeço de Arruda), Sado ~8200 cal BP. Cantabrian: 7981-6636 cal BP (Los Canes, La Braña).",
     "manual_review", "compleja", "en", 3),

    # ═══════════════════════════════════════════════════════════════════
    # PORTUGUÉS — Simples (1)
    # ═══════════════════════════════════════════════════════════════════
    (26,
     "Quando os irmãos Siret escavaram La Bastida de Totana?",
     "Em 1886.",
     "manual_review", "simple", "pt", 1),

    # ═══════════════════════════════════════════════════════════════════
    # PORTUGUÉS — Complejas (1)
    # ═══════════════════════════════════════════════════════════════════
    (27,
     "Cronologia do Bronze Final no Levante peninsular.",
     "Bronze tardio (c.1550/1500-1300/1250 cal BC): Oropesa la Vella, Cabezo Redondo. Bronze Final I (1300-1000): Costamar, Cova d'en Pardo. Bronze Final II (1000-850): Ereta del Castellar, Mola d'Agres. Bronze Final III (850-725): La Vital, Peña Negra I. Idade do Ferro antiga (725-550): Vinarragell, El Molón.",
     "manual_review", "compleja", "pt", 5),

    # ═══════════════════════════════════════════════════════════════════
    # CATALÁN — Simples (1)
    # ═══════════════════════════════════════════════════════════════════
    (28,
     "Quina és la datació més antiga del jaciment neolític de Cingle del Mas Nou?",
     "8007-7583 cal BP.",
     "manual_review", "simple", "ca", 1),

    # ═══════════════════════════════════════════════════════════════════
    # FRANCÉS — Complejas (1)
    # ═══════════════════════════════════════════════════════════════════
    (29,
     "Datations paléolithiques les plus anciennes par communauté autonome en péninsule ibérique.",
     "Andalousie 51.914±45, Aragon 25.330±80, Cantabrie 48.200±80, Castille-et-Léon 30.300±25, Castille-La Manche 28.660±40, Catalogne 38.640±50, C. Madrid 30.280±28, Navarre 21.600±30, C. Valenciana 33.900±60, Estrémadure 61.219±70, Galice 31.690±50, Îles Baléares 24.220±115, La Rioja 6.220±100, Pays Basque 34.350±130, Asturies 16.700±30, Murcie 12.030±0.",
     "manual_review", "compleja", "fr", 10),
]


def get_eval_items():
    """Return list of dicts for Langfuse dataset creation."""
    return [
        {
            "input": {
                "question": item[1],
                "tipo": item[4],
                "idioma": item[5],
                "n_articulos": item[6],
            },
            "expected_output": {
                "ground_truth": item[2],
                "source": item[3],
            },
        }
        for item in EVAL_ITEMS
    ]


def get_metadata():
    """Return dataset statistics."""
    items = EVAL_ITEMS
    return {
        "total": len(items),
        "simples": sum(1 for x in items if x[4] == "simple"),
        "complejas": sum(1 for x in items if x[4] == "compleja"),
        "por_idioma": {
            lang: sum(1 for x in items if x[5] == lang)
            for lang in ["es", "en", "pt", "ca", "fr"]
        },
    }


def get_questions():
    """Return list of question strings."""
    return [item[1] for item in EVAL_ITEMS]


def get_ground_truths():
    """Return list of ground truth strings."""
    return [item[2] for item in EVAL_ITEMS]


def get_sources():
    """Return list of source strings."""
    return [item[3] for item in EVAL_ITEMS]
