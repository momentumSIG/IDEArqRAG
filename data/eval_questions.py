"""RAG-IDEArq-eval-v4 — 32 preguntas canónicas (17 simples + 15 complejas).

Dataset a 27/07/26

Distribución real por idioma:
  Simples   (17): ES 8, EN 5, PT 2, CA 1, FR 1
  Complejas (15): ES 10, EN 1, PT 2, CA 1, FR 1

Cada item: (id, pregunta, ground_truth, fuente, tipo, idioma, n_articulos)
"""

EVAL_ITEMS = [
    # ═══════════════════════════════════════════════════════════════════
    # SIMPLES (17)
    # ═══════════════════════════════════════════════════════════════════
    (0,
     "In what year did the Siret brothers excavate the La Bastida de Totana site?",
     "En 1886.",
     "917_LaBastida_2015.pdf", "simple", "en", 1),

    (1,
     "Which Chalcolithic site is the furthest from the city of Jaén, in the province of Jaén?",
     "The Eras del Alcázar site, situated about 50 km from the city of Jaén.",
     "geojson", "simple", "en", 1),

    (2,
     "Emergency excavations carried out by the Regional Government of Andalusia in the province of Almería, published in 2001 — which sites?",
     "A site on Avenida Pablo Iglesias at the corner of A Rafaela Jiménez, a site on Calle La Central in Villaricos (Cuevas de Almanzora) and on Calle Castillejo (Gádor, Almería).",
     "483_2001_error.pdf", "simple", "en", 1),

    (3,
     "¿Es correcta la datación -1377±23 para Valencina Cerro de la Cabeza Ladera Sur como la más reciente del área?",
     "No, es errónea. La datación correcta más reciente es: Valencina, Cerro de la Cabeza, Ladera Sur, 175 ± 20.",
     "geojson; 1135_Poblado_2021.pdf", "simple", "es", 1),

    (4,
     "¿Estas dataciones son compatibles con el periodo del Neolítico? (Comunidad Valenciana 33.900±60, Galicia 31.690±50, Murcia 12.030±0)",
     "No, estas dataciones son más compatibles con el período Paleolítico y Mesolítico.",
     "1106_Estrategias agropecuarias_2020.pdf", "simple", "es", 1),

    (5,
     "¿Cuál es la fecha más antigua para un yacimiento funerario megalítico en la Península Ibérica?",
     "La fecha más antigua es Trikuaizti 2 (Gipuzkoa), 12015 ± 145.",
     "geojson", "simple", "es", 1),

    (6,
     "Yacimiento con la datación radiocarbónica más antigua de las Islas Baleares.",
     "Cova de Moleta (Sóller), 80000 BP.",
     "geojson", "simple", "es", 1),

    (7,
     "Dataciones más antiguas para el megalitismo en la zona Sureste de la Península Ibérica.",
     "Las primeras manifestaciones funerarias datadas se sitúan en la primera mitad del V milenio con una estimación probabilística para su inicio de entre el 4975-4785 cal BC (68% de probabilidad) y entre el 4740-4570 cal BC (68% de probabilidad) para el final.\n"
     "El análisis de la serie radiocarbónica a partir de diferentes modelados estadísticos permite establecer que el inicio de la actividad funeraria en el denominado “Grupo Puchena” ocurriría entre el 3845-3550 cal BC (95% de probabilidad), posiblemente entre el 3730-3650 cal BC (68% de probabilidad). No parece haber entre los dos tipos de sepulturas analizados diferencias cronológicas significativas en sus inicios. La cronología de rundgräber y sepulturas de cámara y corredor es parecida, quizás ligeramente más antigua en las segundas, aunque las contadas dataciones para estos momentos aconsejan prudencia en este sentido. La primera mitad de IV milenio parece, por tanto, la fecha más probable para el inicio del fenómeno megalítico en el sureste peninsular.",
     "1003_La_cronologia_radiocarbonica_de_las_primeras_manif_2017", "simple", "es", 1),

    (8,
     "Periodización del Bronce Final en el Levante de la Península Ibérica, cronología de las fases y principales ejemplos de yacimientos asignados a las mismas.",
     "Bronce tardío o reciente (c. 1550/1500-1300/1250 cal BC):\n"
     "- Oropesa la Vella\n"
     "- Torreló d'Onda\n"
     "- Les Raboses\n"
     "- Altet de Palau\n"
     "- Cap Prim\n"
     "- Mas del Corral\n"
     "- Cabezo Redondo\n"
     "- Peña de Sax\n"
     "- El Negret\n"
     "- Illeta del Banyets\n"
     "- Tabayá\n"
     "Bronce final I (c. 1300/1250-1000 cal BC):\n"
     "- Costamar\n"
     "- Oropesa la Vella\n"
     "- El Castellet\n"
     "- Torrelló de Boverot\n"
     "- Pic dels Corbs III y IV\n"
     "- Cova d'en Pardo\n"
     "- Cova de la Pastora\n"
     "- Cap Prim\n"
     "- Peña de Sax\n"
     "- El Negret\n"
     "- Tabayá\n"
     "- Botx-Grupitex\n"
     "Bronce final II (1000-850 cal BC):\n"
     "- Ereta del Castellar\n"
     "- El Castellet\n"
     "- Torrelló de Boverot\n"
     "- Pic dels Corbs V\n"
     "- Solana del Castell I\n"
     "- Mola d'Agres\n"
     "- Tabayá\n"
     "- Caramoro\n"
     "- Botx\n"
     "Bronce final III (850-725 cal BC):\n"
     "- Ereta del Castellar\n"
     "- El Castellet\n"
     "- Torrelló de Boverot\n"
     "- Vinarragell\n"
     "- La Vital\n"
     "- Solana del Castell II\n"
     "- Mola d'Agres\n"
     "- Cova de la Sarsa\n"
     "- Tabayá\n"
     "- Peña Negra I\n"
     "- Barranc del Botx\n"
     "- Saladares Ia1/IA2\n"
     "Hierro antiguo o fase Orientalizante (725-550 cal BC):\n"
     "- Vinarragell\n"
     "- El Molón\n"
     "- Los Villares\n"
     "- Solana del Castell III\n"
     "- El Castellar\n"
     "- El Puig\n"
     "- Camara\n"
     "- Tabayá\n"
     "- Peña Negra II\n"
     "- Casa Secà\n"
     "- Saladares IA3",
     "936_BronceFinal_2016.pdf", "simple", "es", 1),

    (9,
     "Cronología y distribución espacial del poblamiento neolítico en la Meseta Sur.",
     "Lo dividimos en dos áreas: poblamiento neolítico en el valle medio y alto del Tajo y poblamiento neolítico de La Mancha.\n"
     "- Valle del Tajo: En la zona de la Sierra madrileña se localiza la Cueva de La Ventana o la Cueva de la Higuera. Ya en la provincia de Guadalajara destacan los yacimientos de la Cueva de la Hoz, Abrigo de Tordelrrábano la Cueva de Jarama II, los enclaves de Sorbe II (Humanes de Mohernando), II y VII, la Cueva del Paso, el Abrigo de los Enebrales, Cueva del Reno y la Cueva del Destete. Sin embargo, son los yacimientos de hoyos los que constituyen el principal modelo de asentamiento. Las excavaciones arqueológicas en extensión realizadas en los últimos años han puesto de manifiesto el predominio del hábitat en asentamientos al aire libre durante este periodo. Tan sólo se han documentado 8 enclaves neolíticos serranos de un total de 24 yacimientos cartografiados en la región. Los 16 yacimientos restantes se ubican en zonas bajas de los valles. A estas cifras hay que sumar tres nuevos yacimientos neolíticos: Soto del Henares, La Serna y Prado de Galápagos, recientemente publicados por C. Blasco et al. (2016) que amplían el listado de asentamientos de hoyos ubicados en los fondos de valle.\n"
     "La concentración de yacimientos neolíticos es especialmente interesante en la zona sureste de la Comunidad de Madrid, en los tramos finales de los ríos Henares, Jarama y Manzanares. La densidad de yacimientos de hoyos neolíticos en las zonas bajas de estos cursos fluviales ha aportado desde hace años datos muy interesantes para conocer los patrones de asentamiento y los modelos de ocupación territorial de las primeras comunidades neolíticas.\n"
     "- Poblamiento neolítico de La Mancha: El modelo de asentamiento neolítico mejor conocido en La Mancha son las ocupaciones en cuevas y abrigos. En este sentido, los yacimientos neolíticos manchegos mejor conocidos son la Cueva del Niño y el Abrigo de Molino de Vadico, en Albacete, y el Abrigo de Verdelpino, en Cuenca, a los que ya nos referimos en el capítulo anterior. En estos yacimientos se han documentado, además, ocupaciones previas, por lo que se son especialmente interesantes para estudiar el momento de adopción de los modos de vida neolíticos.\n"
     "Si bien es cierto que las investigaciones sobre esta etapa de la Prehistoria en La Mancha han sido escasas y son pocos los equipos de investigación que han desarrollado líneas de investigación orientadas al conocimiento de las primeras sociedades productoras en esa región, no es menos cierto que las intervenciones arqueológicas llevadas a cabo en el contexto de obras de infraestructura en Castilla-La Mancha apoyan la idea de un poblamiento neolítico caracterizado por su escasa densidad, especialmente si se compara con otras regiones como en valle medio y alto del Tajo.",
     "984_DemografiayPoblamiento_2016.pdf", "simple", "es", 1),

    (10,
     "Principales yacimientos de la Segunda Edad del Hierro en la provincia de León.",
     "Los principales yacimientos de la Segunda Edad del Hierro en la provincia de León son los siguientes:\n"
     "- Castro de la Edad de Hierro de Valencia de Don Juan\n"
     "- La Muela (al otro lado de la carretera de acceso a Valencia de Don Juan)\n"
     "- Antigua ciudad astur-romana de Lancia.\n"
     "- Regueras de Arriba o San Martín de Torres.\n"
     "- En la zona cántabra, los castros laciniaegos de la Mesa en Rioscuro, que se delimita con una potente muralla de módulos, fechada seguramente en los siglos II y I a.C., y el castro de La Zamora, en sosas de Laciana.\n"
     "- También en la zona cántabra en el municipio de Puebla de Lillo se han podido reconocer ciertas piezas metalíticas de la Segunda Edad del Hierro, recogidas en el antiguo castro de Castiltejón.\n"
     "- El castro de Chano en la comarca de Fornela.\n"
     "- Peña del Castro (La Ercina), ubicado a 2 km de la localidad de la Ercina\n"
     "- El Castrelín de San Juan dse Paluezas\n"
     "- La Corona del Castro en Borrenes\n"
     "- La Peña del Hombre en el ayuntamiento de Priaranza\n"
     "- Castro de Columbrianos\n"
     "- Peña Piñera, en la Vega de Espinareda\n"
     "Por último, se ha señalado un conjunto de castros en altitudes considerables en las sierras del Teleno, la Valdería y el Bierzo, algunos ya conocidos en la bibliografía y en la Carta Arqueológica de León, con grandes amurallamientos, que se extienden entre afloramientos rocosos de materiales de la era Primaria; yacimientos como Portillo de Xandequín en Pozos, Peña Rayada en Cunas, Alto de San Vicente-Los Conventos en Morla de la Valdería, Yera de los Piornos-Peña del Tren en Torneros de la Valdería, Sierra del Pueblo en Torneros de la Valdería, El Pajarín-La Formosida en Boisán (Lucillo) y los bercianos localizados en Folgoso de la Ribera, Torre del Bierzo y Molinaseca.",
     "921_CelisSanchezMuozVillarejo-2015-Veinteaosdeinvestigaciones.pdf", "simple", "es", 1),

    (11,
     "Quins tipus de ceràmica s'han trobat a la Sima del Ciervo II (Gabasa, Peralta de Calasanz, Osca) i a quina fase del període neolític pertanyen?",
     "La Sima del Ciervo II (Gabasa, Peralta de Calasanz, Osca) és una cova en gran part vertical que ha proporcionat un conjunt ceràmic atribuïble al neolític inicial, concretament a la facies epicardial, amb una cronologia aproximada del 5300–4800 cal aC.\n"
     "Els recipients presenten decoracions que combinen tècniques d'incisió i impressió, impressions d'eines (simples, de doble punta o múltiples), ranuració i decoracions marcades amb corda. Aquestes característiques corresponen a les de la facies epicardial del neolític inicial, que està molt ben representada a les zones del Cinca i el Segre.",
     "1162_La sima del ciervo_2022.pdf", "simple", "ca", 1),

    (12,
     "Quels sont les principaux modèles théoriques de l'expansion du Néolithique en Europe?",
     "On peut les classer en deux grandes catégories: la première, connue sous le nom de « diffusion démique », met l’accent sur les mouvements des sociétés néolithiques et, par extension, sur les pratiques agricoles ; la seconde, appelée « diffusion culturelle » dans la littérature spécialisée, souligne l’importance de la transmission de l’ensemble des éléments caractéristiques du Néolithique — technologies (par exemple, la poterie), plantes et animaux domestiques — en tant que facteur déclencheur de l’expansion du Néolithique.",
     "904_SalvadorPardo_2015.pdf", "simple", "fr", 1),

    (13,
     "Quais são as datas mais antigas da extração de sílex na península central?",
     "A única mina de sílex do Neolítico no centro de Espanha é a de Casamontero (Madrid). Infelizmente, a amostra de Sus sp. não tinha colagénio suficiente para ser datada. O teste X2 mostra que todas as datas, com a única exceção da Beta-232890, são estatisticamente idênticas. Recentemente, foi enviada outra matriz de anel para datação num fragmento de fémur de Ovis aries. O resultado, 6200+/ -40 BP (Beta-295152), é estatisticamente igual a dez das onze datas anteriores.\n"
     "Assumindo a hipótese plausível de que elas datam diferentes eventos de mineração, há uma probabilidade de 65% de que todos os episódios de mineração tenham ocorrido entre 5327-5215 cal BC (1σ), um período de tempo de apenas cem anos. Portanto, estas datas de radiocarbono não permitem observar a evolução espácio-temporal da exploração mineira, mas indicam que o principal episódio de atividade da Casa Montero durou pouco mais de um século, quatro gerações. Esta interpretação não é apenas possível, é provável.",
     "903_La_cronologia_absoluta_de_la_mineria_2015.pdf", "simple", "pt", 1),

    (14,
     "Sitios arqueológicos calcolíticos da Península Ibérica onde foram encontrados objetos de marfim.",
     "Calcolítico antigo (pré-campaniforme [bell beaker]):\n"
     "- Zambujal\n"
     "- Vila Nova de São Pedro\n"
     "- Leceia\n"
     "- Praia das Maçãs\n"
     "- Palmela\n"
     "- Alcalar\n"
     "- Perdigões\n"
     "- Señorío de Guzmán\n"
     "- La Pijotilla\n"
     "- Valencina de la Concepción\n"
     "- Gilena\n"
     "- Los Millares\n"
     "Calcolítico recente (campaniforme [bell beaker]):\n"
     "- Palmela\n"
     "- Pedra do Ouro\n"
     "- Verdelha dos Ruivos\n"
     "- Vila Nova de São Pedro\n"
     "- Perdigões\n"
     "- Valencina de la Concepción\n"
     "- Los Algarbes\n"
     "- Cerro de la Virgen\n"
     "- Camino de Yeseras\n"
     "- La Pijotilla",
     "993_Bartelheim_et_al_Iberian Chalcolithic_2017.pdf", "simple", "pt", 1),

    (15,
     "What evidence of agriculture and animal domestication has been documented at the Neolithic settlement of La Draga (Banyoles, Girona)?",
     "• Agriculture: The cultivation of naked wheat (predominant), hulled barley, and hulled wheat varieties such as farro and spelt has been documented. The fields were small and worked by hand using digging sticks, with very intensive weeding practised. The grain was stored in huts. Naked wheat was stored after threshing, whereas barley and hulled wheats were stored unhusked. There is also evidence of agricultural technology (artefacts related to agriculture, such as parts of sickles, hoes, etc.).\n"
     "• Livestock: The domesticated species identified are pigs, cattle, goats, sheep and dogs. Mixed livestock farming was practised, combining herds of the four species (pigs, cattle, goats and sheep). Domesticated species accounted for over 97 per cent of the faunal remains, indicating a predominantly livestock-based economy. Hunting (aurochs, red deer, wild boar, roe deer, wild goat) was supplementary, and dogs were probably used for guarding livestock and hunting. Pigs were mainly reared for meat and, to a lesser extent, for raw materials. Animals of this species were slaughtered for consumption when they were around one year or one and a half years old. As for goats and cows, there is evidence of a significant culling of young animals, a practice that can be linked to the use of the females’ milk, from which dairy products could be obtained. Bone pathologies have been documented in some cow specimens, likely caused by their use for transport. Like oxen, goats and pigs, sheep were reared primarily for meat production. Individuals over four years of age have been documented, suggesting the possibility of wool production, although there is no evidence of fibres.\n"
     "• Subsistence: The diet was varied, combining domestic and wild products, although the bulk of the diet came from domesticated species.",
     "1006_La revolució neolítica. La Draga, el poblat dels prodigis_2017.pdf", "simple", "en", 1),

    (16,
     "When did human settlements in the high mountains of the Pyrenees begin, and which archaeological sites provide evidence of them?",
     "Las ocupaciones humanas en alta montaña en los Pirineos se documentan desde el Mesolítico (12.000 cal BC) hasta el Calcolítico (2000 cal BC):\n"
     "• Mesolítico: Ocupaciones estacionales en abrigos y cuevas de media montaña, así como en abrigos y a acielo abierto en zonas altas, con evidencias en ambo casos de caza y recolección. Tenemos los yacimientos del nivel 5 del Dolmen de la Font dels Coms, en la cabecera del valle de Baiasca, yacimiento de Orrís de Perafita II, en Andorra, y el tercer yacimiento con restos anteriores al proceso de neolitización en los Pirineos es el Abric de l’Estany de la Coveta I.\n"
     "• Neolítico Antiguo (segunda mitad del VI milenio cal BC): Yacimientos como Coro Tracito, Els Trocs y El Sardo confirman la presencia de comunidades agropastorales por encima de los 1500 m. Además, lael yacimiento de Balma Margineda (Andorra), que actualmente se interpreta como parte de una ocupación larga de media montaña desde desde 12.000 cal BC hasta 5000 cal BC. Esta cueva tiene ocupación durante el Paleolítico y Mesolítico, además de algunas dataciones post-neolíticas.\n"
     "• Neolítico final / Calcolítico: Intensificación del poblamiento con pastoralismo organizado y estructuras (cabañas, almacenamientos) hasta 2400 m",
     "1025_The_Beginning_of_High_Mountain_Occupations_in_the_2017.pdf", "simple", "en", 1),

    # ═══════════════════════════════════════════════════════════════════
    # COMPLEJAS (15)
    # ═══════════════════════════════════════════════════════════════════
    (17,
     "Quines són les cronologies de les manifestacions funeràries del Mesolític a les diferents regions peninsulars?",
     "La diferència més significativa entre la regió mediterrània i la regió cantàbrica i Portugal s'observa en l'aparició de cementiris durant el mesolític inicial. A més, en avaluar les dades de les tres zones de la Península Ibèrica, es poden identificar els següents patrons cronològics.\n"
     "A la regió mediterrània, les datacions d'El Collado mostren que els cementiris van aparèixer al voltant del 9475-9300 cal BP. Aquest tipus de pràctica funerària continuarà en altres jaciments propers, com Casa Corona i Cingle del Mas Nou. A diferència del cementiri d'El Collado, que va estar en ús durant uns 1.100 anys, segons les datacions obtingudes, a Casa Corona i Cingle del Mas Nou el seu període d'ús és molt més breu (8007-7583 cal BP). A més, Cingle del Mas Nou es diferencia d'altres jaciments funeraris mesolítics de la Península Ibèrica en què els restes de set individus (complets i incomplets) van ser dipositats en una única estructura.\n"
     "A la façana atlàntica de Portugal, les primeres proves de cementiris a l'estuari de Muge daten de 8409-8030 cal BP (a Cabeço de Arruda, per exemple). Aquests s'associen a grans jaciments de restes de mol·luscs de més de 5 m de gruix, utilitzats durant un llarg període de temps. A l'estuari del Sado, les datacions són lleugerament més recents que a Muge, començant al voltant del 8200 cal BP (per exemple, a Amoreiras). Està clar que entre el 8160 i el 7970 cal BP, els grups mesolítics enterraven sistemàticament tots o alguns dels seus morts en cementiris.\n"
     "Finalment, les datacions dels jaciments funeraris del nord de la Península Ibèrica (costa cantàbrica) amb dos o més individus indiquen que els primers enterraments mesolítics agrupats van ser una mica més recents (entre 7981 i 6636 cal BP). En qualsevol cas, cal destacar que, a diferència de les altres dues zones, en la majoria dels jaciments només s'ha documentat un únic individu o grups molt més reduïts, com en Los Canes i La Braña.",
     "1168_The mesolithic of atlantic coastal spain_2022.pdf; 919_JuanGibaja_2015.pdf", "compleja", "ca", 2),

    (18,
     "Dataciones más antiguas (i.e, más altas) de yacimientos paleolíticos para cada comunidad autónoma.",
     "Las dataciones son las siguientes:\n"
     "- Andalucía, 51.914 ± 45\n"
     "- Aragón, 25.330 ± 80\n"
     "- Cantabria, 48.200 ± 80\n"
     "- Castilla y León, 30.300 ± 25\n"
     "- Castilla-La Mancha, 28.660 ± 40\n"
     "- Cataluña, 38.640 ± 50\n"
     "- Comunidad de Madrid, 30.280 ± 28\n"
     "- Comunidad Foral de Navarra, 21.600 ± 30\n"
     "- Comunitat Valenciana, 33.900 ± 60\n"
     "- Extremadura, 61.219 ± 70\n"
     "- Galicia, 31.690 ± 50\n"
     "- Illes Balears, 24.220 ± 115\n"
     "- La Rioja, 6.220 ± 100\n"
     "- País Vasco, 34.350 ± 130\n"
     "- Principado de Asturias, 16.700 ± 30\n"
     "- Región de Murcia, 12.030 ± 0",
     "varias entradas del geojson", "compleja", "es", 1),

    (19,
     "Yacimientos neolíticos situados a menos de 150 km de Casa Montero.",
     "Los yacimientos neolíticos situados a menos de 150 km de Casa Montero son los siguientes:\n"
     "- La Atalaya, situado en la provincia de Ávila (Muñopepe). Coordenadas: longitud -4,817, latitud 40,636.\n"
     "- Portillo de las Cortes, situado en la provincia de Guadalajara (Anguita). Coordenadas: longitud -2,407, latitud 41,056.\n"
     "- El Cañaveral, situado en la provincia de Madrid (Madrid). Coordenadas: longitud -3,56, latitud 40,407.\n"
     "- Pista de Motos, situado en la provincia de Madrid (Madrid). Coordenadas: longitud -3,665, latitud 40,339.\n"
     "- O'Donnell, situado en la provincia de Madrid (Madrid). Coordenadas: longitud -3,658, latitud 0,419.\n"
     "- Casa Montero, situado en la provincia de Madrid (Madrid). Coordenadas: longitud -3,5257, latitud 40,4049.\n"
     "- Colector H05, situado en la provincia de Madrid (Madrid). Coordenadas: longitud -3,671, latitud 40,351.\n"
     "- Cueva de la Higuera, situado en la provincia de Madrid (Patones). Coordenadas: longitud -3,509, latitud 40,854.\n"
     "- Capanegra/Deseada, situado en la provincia de Madrid (Rivas-Vaciamadrid). Coordenadas: longitud -3,52, latitud 40,355.\n"
     "- El Congosto, situado en la provincia de Madrid (Rivas-Vaciamadrid). Coordenadas: longitud -3,556, latitud 40,329.\n"
     "- Cueva de la Ventana, situado en la provincia de Madrid (Torrelaguna). Coordenadas: longitud -3,527, latitud 40,851.\n"
     "- Cueva de la Vaquera, situado en la provincia de Segovia (Torreiglesias). Coordenadas: longitud -4,059, latitud 41,086.\n"
     "- La Mina, situado en la provincia de Soria (Alcubilla de las Peñas). Coordenadas: longitud -2,55, latitud 41,25.\n"
     "- Túmulo de la Sima, situado en la provincia de Soria (Miño de Medinaceli). Coordenadas: longitud -2,54, latitud 41,177.\n"
     "- Peña de la Abuela, situado en la provincia de Soria (Miño de Medinaceli). Coordenadas: longitud -2,514, latitud 41,157.\n"
     "- La Tarayuela, situado en la provincia de Soria (Miño de Medinaceli). Coordenadas: longitud -2,503, latitud 41,167.\n"
     "- La Revilla del Campo, situado en la provincia de Soria (Miño de Medinaceli). Coordenadas: longitud -2,515, latitud 41,17.\n"
     "- Carlos Álvarez, situado en la provincia de Soria (Miño de Medinaceli). Coordenadas: longitud -2,546, latitud 41,187.\n"
     "- La Lámpara, situado en la provincia de Soria (Miño de Medinaceli). Coordenadas: longitud -2,515, latitud 41,156.\n"
     "- El Castillejo, situado en la provincia de Toledo (Huecas). Coordenadas: longitud -4,2092, latitud 39,9914.\n"
     "- El Tonto, situado en la provincia de Toledo (Mocejón). Coordenadas: longitud -3,931, latitud 39,944.\n"
     "- La Paleta, situado en la provincia de Toledo (Numancia de la Sagra). Coordenadas: longitud -3,864, latitud 40,085.\n"
     "- La Ocañuela, situado en la provincia de Toledo (Ocaña). Coordenadas: longitud -3,56196, latitud 39.95578\n",
     "varias entradas del geojson", "compleja", "es", 1),

    (20,
     "Publications présentant des données paléogénétiques issues de sites archéologiques de la péninsule ibérique.",
     "Les publications du corpus qui présentent des données paléogénétiques issues de sites de la péninsule sont les suivantes :\n"
     "- Szécsényi-Nagy et al. 2017 — « The maternal genetic make-up of the Iberian Peninsula between the Neolithic and the Early Bronze Age » (Scientific Reports 7:15644).\n"
     "Cette étude analyse l’ADN mitochondrial de 140 individus, du Néolithique à l’âge du bronze, sur la péninsule ibérique. Elle identifie des haplogroupes maternels et retrace la continuité génétique entre les populations néolithiques et chalcolithiques.\n"
     "- Olalde et al. 2018 — «Le phénomène Beaker et la transformation génomique du nord-ouest de l’Europe » (Nature).\n"
     "Présente des données d’ADN ancien (aDNA) provenant de plus de 400 individus européens, y compris des sites péninsulaires du Chalcolithique (culture des vases campaniformes). Démontre que l’arrivée de la culture des vases campaniformes a été associée à un remplacement génétique massif (~40 %) dans la péninsule ibérique.\n"
     "- Palomo-Díez et al. 2017 — «Un cas inattendu dans la préhistoire de la péninsule ibérique : analyse de l’origine biogéographique à partir de l’ADN mitochondrial » (Forensic Science International: Genetics).\n"
     "Détermine l’haplotype mitochondrial N9a2'4'5 d’un individu de l’âge du bronze en péninsule ibérique, avec une analyse de l’origine biogéographique.\n"
     "- Gurke et al. 2021 — « Aperçu de la domestication des bovins et du processus de néolithisation en Galice (Espagne) à la lumière de données génétiques » (PLOS ONE). Séquençage du génome mitochondrial de 11 bovins préhistoriques découverts dans des grottes galiciennes, permettant d'identifier les haplogroupes T et I afin de reconstituer l'introduction du bétail au Néolithique.",
     "1026_Material genetic_2017.pdf; 1062_The beaker phenomenon_2018.pdf; 977_UnexpectedCase_2017.pdf; 1138_Insight_2021.pdf", "compleja", "fr", 4),

    (21,
     "Publications dating from after 2010 presenting carbon-14 dates from Neolithic sites on the Iberian Peninsula.",
     "The publications in the corpus (all dating from after 2010) that present carbon-14 dates from Neolithic sites on the Iberian Peninsula are as follows:\n"
     "Gibaja et al. 2016 — 'Human diet and the chronology of Neolithic societies in the north-east of the Iberian Peninsula: the necropolises of Puig d'en Roca and Can Gelats (Girona, Spain)' (Archaeological and Anthropological Sciences). This article provides AMS dates for human bones from the necropolises of Puig d'en Roca and Can Gelats (Girona), with dates such as 5095 ± 35 BP (3968–3798 BC cal.) for the Middle-Late Catalan Neolithic.\n"
     "Aranda Jiménez et al. 2017 — 'The radiocarbon chronology of the earliest megalithic sites in the south-east of the Iberian Peninsula' (Trabajos de Prehistoria 74: 257–277). This article presents radiocarbon dates from the megalithic necropolises of Las Churuletas, La Atalaya and Llano del Jautón (Purchena, Almería), using Bayesian modelling to establish the chronology of Neolithic–Chalcolithic megalithism in the south-east.\n"
     "García Rivero 2022 — 'Cueva de la Dehesilla: new stratigraphic, material and radiocarbon data (trench c006) and a review of its Neolithic sequence' (SPAL 31). This article provides new radiocarbon dates for the Neolithic sequence from the Dehesilla cave (Cádiz), thereby revising the chronology of the Neolithic period in the south of the peninsula.\n"
     "Díaz-Zorita Bonilla et al. 2021 — 'Neolithic networks and mobility during the 5th and 4th millennia'.",
     "1002_gibaja2016.pdf; 1003_La_cronologia_radiocarbonica_de_las_primeras_manif_2017.pdf; 1153_Cueva de la dehesilla_2022.pdf; 1141_Neolithic networking_2021.pdf; 965_XavierOms_2016.pdf", "compleja", "en", 5),

    (22,
     "Que evidências estratégias de gestão florestal e de obtenção de recursos madeireiros foram documentadas nas comunidades do Bronze Inicial e Médio do noroeste atlântico da Península Ibérica?",
     "As comunidades do Bronze Inicial e Médio do noroeste atlântico da Península Ibérica exploravam principalmente os recursos florestais disponíveis nas imediações dos povoados, com um claro predominância do carvalho caducifólio (Quercus sp.). As análises antracológicas documentam também a exploração de espécies como o aveleiro (Corylus avellana), o freixo (Fraxinus sp.), Prunus sp., o salgueiro/choupo (Salix/Populus), o medronheiro (Arbutus unedo), a azinheira (Quercus perennifolio) e diversas Fabaceae e Rosaceae/Maloideae. O castanheiro não pode ser identificado de forma independente, uma vez que aparece agrupado com Quercus nas análises (Quercus/Castanea), enquanto o pinheiro (Pinus pinea/pinaster) teve uma presença muito limitada e restrita principalmente às zonas costeiras. A exploração florestal desenvolveu-se em diferentes biótopos — especialmente florestas mistas caducifólias, florestas ribeirinhas e matagais —, observando-se um aumento progressivo do matagal ao longo da Idade do Bronze. As evidências indicam uma gestão ativa da paisagem através do abate seletivo, do uso recorrente do fogo para a gestão florestal e de uma pressão contínua sobre os carvalhais, favorecendo a regeneração de determinadas espécies e a criação de clareiras. As espécies lenhosas desempenhavam múltiplas funções. O carvalho fornecia madeira de construção e combustível, enquanto as Fabaceae forneciam lenha, materiais leves para construção, ferramentas e cestaria.",
     "1023_Socio_ecology_2017.pdf; 1022_Monteroetal4AiE_2017.pdf; 1054_Landscapes_2018.pdf", "compleja", "pt", 3),

    (23,
     "Que diferenças tipológicas existem entre a cerâmica Cardial e a Epicardial?",
     "Cardial pottery (6th millennium BC) is characterised primarily by decoration created using impressions from cockle shells (Cardium edule, now known as Cerastoderma edule), which constitutes its diagnostic feature. In terms of forms, the predominant types are non-standardised vessels with simple profiles — hemispherical, sub-spherical or cylindrical — generally with a globular body and a rounded base, whilst necked vessels represent the main documented composite form.\n"
     "Epicardial pottery (second half of the 6th millennium cal BC) replaces cardial impressions with a greater diversity of decorative techniques, such as incisions, impressions made with tools, grooving and applied cord marks. These differences reflect an evolution of decorative traditions compared to the Cardial. In the corpus analysed, information on Epicardial pottery forms is more limited; consequently, the main documented differences between the two types of production relate to decorative techniques.\n"
     "In the available corpus, the best-documented differences are decorative; typological differences relating to pottery forms are described to a much more limited extent.",
     "1162_La sima del ciervo_2022.pdf; 1091_Vaso decoracion_2019.pdf", "compleja", "pt", 2),

    (24,
     "¿Qué evidencias arqueológicas documentan el proceso de neolitización en la Península Ibérica?",
     "La neolitización de la Península Ibérica se documenta mediante un conjunto de evidencias arqueológicas que reflejan la adopción progresiva de nuevas formas de producción, tecnología y organización social, aunque su cronología y ritmo varían según las regiones.\n"
     "Desde el punto de vista arqueobotánico, se constata la introducción de cultivos domesticados, principalmente diferentes variedades de trigo (Triticum), cebada (Hordeum) y leguminosas como el guisante (Pisum sativum). En el ámbito zooarqueológico, aumenta el predominio de especies domésticas, alcanzando porcentajes superiores al 97 % en yacimientos como La Draga, lo que evidencia una economía plenamente agropecuaria.\n"
     "La neolitización también se manifiesta en importantes cambios en la cultura material, entre ellos la aparición y generalización de la cerámica, el desarrollo del tejido y la cestería, la introducción de la industria lítica pulimentada (hachas y azuelas), la incorporación de útiles especializados como los dientes de hoz y un notable desarrollo del trabajo de la madera, bien documentado en La Draga. Asimismo, se observa un incremento de las prácticas funerarias, especialmente de carácter colectivo.\n"
     "Desde el punto de vista cronológico y espacial, el proceso no fue simultáneo en toda la Península. Las evidencias del corpus consultado sitúan asentamientos plenamente neolíticos en el noreste entre 5600 y 5300 cal BC y muestran que la transición fue gradual en algunas áreas pirenaicas. No obstante, el conjunto de la investigación arqueológica indica que la neolitización comenzó antes en las costas mediterráneas, Andalucía y el litoral atlántico portugués, extendiéndose posteriormente hacia el interior y alcanzando de forma más tardía el norte y noroeste peninsular, en un proceso desarrollado a lo largo de aproximadamente dos milenios (VI-IV milenios cal BC).",
     "1006_La revolució neolítica_2017.pdf; 1025_High_Mountain_Occupations_2017.pdf; 1000_oms_2017.pdf; 1016_Cova_Guineu_2017.pdf; 1006_La revolució neolítica. La Draga, el poblat dels prodigis_2017.pdf", "compleja", "es", 5),

    (25,
     "¿Cuáles son los principales tipos de arquitectura megalítica documentados en el norte peninsular y cómo se distribuyen geográficamente?",
     "En el norte peninsular se documentan distintos tipos de arquitectura megalítica, tanto funeraria como no funeraria.\n"
     "Entre las construcciones funerarias destacan los dólmenes, formados por una cámara construida con grandes ortostatos y cubierta por una o varias losas, generalmente bajo un túmulo, utilizados como enterramientos colectivos durante el Neolítico y el Calcolítico. Los sepulcros de corredor constituyen una variante de los dólmenes, caracterizada por la presencia de un pasillo que comunica la cámara con el exterior. También se documentan cistas, de menor tamaño y normalmente destinadas a enterramientos individuales o de pocos individuos, más frecuentes durante la Edad del Bronce. Aunque no constituyen arquitectura megalítica en sentido estricto, las cuevas sepulcrales (naturales o artificiales) suelen considerarse dentro del mismo fenómeno funerario por corresponder igualmente a prácticas de enterramiento colectivo del Neolítico y el Calcolítico.\n"
     "El megalitismo no funerario está representado por los menhires, grandes piedras hincadas verticalmente, y por los crómlechs, estructuras circulares formadas por piedras verticales, cuya cronología se sitúa principalmente entre la Edad del Bronce y la Edad del Hierro.\n"
     "Desde el punto de vista de su distribución, los dólmenes y sepulcros de corredor aparecen ampliamente repartidos por el norte peninsular. Las cuevas sepulcrales son especialmente frecuentes en el sector oriental (Navarra, País Vasco y Cantabria), favorecidas por la presencia de relieves calizos. Las cistas también se documentan en numerosas áreas del norte, mientras que menhires y, especialmente, crómlechs presentan una mayor concentración en la zona oriental del norte peninsular.",
     "1168_The mesolithic of atlantic coastal spain_2022.pdf; 943_DolmenHuesera_FernandezEraso_2016.pdf; 757_2011.pdf; 1150_At the begginings_2022.pdf; 1146_Subsistence.pdf", "compleja", "es", 5),

    (26,
     "¿Qué evidencias arqueológicas documentan la producción metalúrgica durante la Edad del Bronce en el levante peninsular?",
     "Las evidencias arqueológicas documentan una actividad metalúrgica consolidada en el levante peninsular durante la Edad del Bronce (ca. 2200–1800 cal BC), basada tanto en la producción local de objetos metálicos como en el intercambio de metales y piezas de mayor prestigio.\n"
     "En yacimientos como Santa Llúcia (Alcalà de Xivert-Alcossebre, Castellón) se han recuperado evidencias directas de producción metalúrgica, entre ellas gotas de fundición o reducción, un lingote en forma de barra, un remache, una lámina doblada y una daga con tres remaches,\n"
     "hallados asociados a una vasija. Estos materiales indican actividades de transformación y manufactura de objetos metálicos a pequeña escala.\n"
     "El estudio de estos materiales mediante análisis elementales, metalografía e isótopos de plomo ha permitido identificar un modelo dual de producción y abastecimiento. Por un lado, existía una producción doméstica o local, destinada principalmente a fabricar pequeños útiles, como punzones o remaches, utilizando recursos regionales e incluso metal reciclado. Por otro, determinados objetos de mayor tamaño o prestigio, como algunas alabardas, parecen haberse obtenido mediante redes de intercambio de ámbito suprarregional, lo que evidencia la circulación de materias primas y objetos metálicos entre distintas comunidades.\n"
     "Además de la metalurgia del bronce, durante este período también se documenta el trabajo del oro, como muestran los dos tútuli de oro —adornos de suspensión indirecta— recuperados en el levante peninsular, que evidencian la fabricación o circulación de objetos de carácter ornamental y probablemente de prestigio.",
     "1022_Monteroetal4AiE_2017.pdf; 1108_Cronologia_fundacional_2020.pdf; 1013_Objetos_de_oro_2017.pdf", "compleja", "es", 3),

    (27,
     "¿Qué tipos de enterramientos colectivos se documentan durante el Neolítico y el Calcolítico en la Península Ibérica?",
     "Las prácticas funerarias colectivas están documentadas en la Península Ibérica desde el Neolítico, aunque alcanzan su mayor desarrollo durante el Neolítico Final y el Calcolítico, cuando constituyen uno de los rasgos característicos de estas comunidades. Entre las evidencias más antiguas se encuentran algunos enterramientos colectivos en cueva, como Cova Bonica (Barcelona). Asimismo, se conocen necrópolis neolíticas, como las de Puig d'en Roca y Can Gelats (Girona), que evidencian la existencia de espacios específicamente destinados a usos funerarios. A partir del Neolítico Final y durante el Calcolítico, los enterramientos colectivos se generalizan en gran parte de la Península Ibérica mediante dos grandes tipos de contextos funerarios: los monumentos megalíticos, como dólmenes y sepulcros de corredor, y las cuevas sepulcrales, ambos destinados a la inhumación sucesiva de múltiples individuos. Estas prácticas se caracterizan por la deposición reiterada de individuos en un mismo espacio funerario y por la presencia habitual de ajuares y ofrendas, además de evidencias de actividades rituales asociadas. En conjunto, reflejan una concepción comunitaria del espacio funerario y constituyen uno de los rasgos culturales más representativos de las sociedades del Neolítico Final y del Calcolítico en la Península Ibérica.",
     "1000_oms_2017.pdf; 1070_Deciphering Neolithic_2019.pdf; 1002_gibaja2016.pdf", "compleja", "es", 3),

    (28,
     "¿Qué papel desempeñó la caza de grandes mamíferos en las comunidades de la Prehistoria Reciente de la Península Ibérica?",
     "Durante la Prehistoria Reciente (Neolítico, Calcolítico y Edad del Bronce), la caza de grandes mamíferos continuó practicándose en la Península Ibérica, aunque desempeñó un papel complementario dentro de economías basadas principalmente en la agricultura y la ganadería. Su importancia fue mayor durante el Neolítico y tendió a disminuir progresivamente a lo largo del Calcolítico y la Edad del Bronce.\n"
     "Las evidencias arqueozoológicas documentan la explotación de diversas especies silvestres, entre las que destacan el ciervo (Cervus elaphus), el jabalí (Sus scrofa), el corzo (Capreolus capreolus) y la cabra montés (Capra pyrenaica). De forma menos frecuente también se registran uro (Bos primigenius) y caballo salvaje (Equus caballus). Además, algunos yacimientos proporcionan evidencias de la caza ocasional de grandes carnívoros, como lobo (Canis lupus), zorro (Vulpes vulpes) y oso pardo (Ursus arctos).\n"
     "Un ejemplo destacado es La Draga (Banyoles, Girona), donde se han documentado restos de uro, ciervo y jabalí. En el caso del uro, se ha estimado un rendimiento de hasta 300 kg de carne, lo que pone de manifiesto el importante aporte puntual que podía representar la captura de grandes herbívoros. Estas evidencias indican que la caza siguió siendo una actividad relevante para complementar la dieta y el aprovechamiento de recursos animales, aunque subordinada a la producción agropecuaria.",
     "1006_La revolució neolítica_2017.pdf; 1000_oms_2017.pdf; 1016_Cova_Guineu_2017.pdf; 1025_High_Mountain_2017.pdf", "compleja", "es", 4),

    (29,
     "¿Cómo se organizaron las redes de intercambio de jadeíta y variscita durante el Neolítico y el Calcolítico en la Península Ibérica?",
     "Durante el Neolítico y el Calcolítico (V–III milenios cal BC), la jadeíta y la variscita formaron parte de amplias redes de intercambio que conectaban diferentes regiones de Europa y la Península Ibérica. Ambos materiales fueron valorados no solo por sus propiedades físicas, sino también por su fuerte carga simbólica, asociada probablemente al prestigio, la fertilidad, la vida y el poder.\n"
     "• Jadeíta. Las hachas de jadeíta procedían de los Alpes y alcanzaron numerosos yacimientos peninsulares, especialmente en el noreste. Su distribución parece haberse articulado mediante dos grandes rutas: una ruta mediterránea, que seguía la costa francesa hasta los centros de Gavà y Cardona, desde donde las piezas se redistribuían hacia el valle del Ebro, Levante y Andalucía; y una ruta atlántica, que conectaba Bretaña con la cornisa cantábrica, Galicia, la Meseta Norte y Portugal.\n"
     "• Variscita. La variscita se explotó principalmente en Can Tintorer (Gavà, Barcelona), Palazuelo de las Cuevas (Zamora) y Pico Centeno (Encinasola, Huelva) para fabricar cuentas de collar y colgantes. La mayoría de estos objetos procede de contextos funerarios, especialmente de monumentos megalíticos como los dólmenes de corredor. Su producción y circulación alcanzaron su máxima intensidad entre finales del IV y el III milenio cal BC. Los análisis de procedencia indican que las variscitas orientales y occidentales siguieron circuitos de distribución diferenciados, separados en términos generales por la divisoria entre las cuencas mediterránea y atlántica. Además, la elevada concentración de cuentas en zonas como el estuario del Tajo sugiere la existencia de importantes centros de redistribución, mientras que la proximidad a las minas no implicaba necesariamente un mayor acceso a estos adornos.\n"
     "La frecuente asociación de hachas de jadeíta y adornos de variscita en algunos ajuares funerarios, especialmente en Cataluña y la Bretaña francesa, indica que ambos materiales circularon, al menos parcialmente, a través de las mismas redes de intercambio.",
     "Variscita_jadeita_2021.pdf; 932_distribution-and-chronological-framework-for-iberian-variscite-mining_2016.pdf", "compleja", "es", 2),

    (30,
     "¿Cómo evolucionaron las redes de intercambio de ámbar y marfil durante el Calcolítico y la Edad del Bronce en la Península Ibérica?",
     "Durante el Calcolítico y la Edad del Bronce, las redes de intercambio de ámbar y marfil conectaron la Península Ibérica con regiones muy alejadas del Mediterráneo y Europa, favoreciendo la circulación de materias primas exóticas destinadas principalmente a la elaboración de objetos de prestigio y adorno.\n"
     "• Ámbar. Desde el IV milenio cal BC, predominó el ámbar de origen siciliano (simetita), como demuestra la cuenta recuperada en la Cueva de los Cuarenta (Priego de Córdoba), identificada mediante espectroscopía FTIR. A partir del último cuarto del II milenio cal BC, el ámbar procedente del Báltico fue sustituyendo progresivamente al siciliano, reflejando una reorganización de las redes de intercambio europeas.\n"
     "• Marfil. El marfil tuvo una especial importancia durante el Calcolítico y la Edad del Bronce, especialmente en el sur de la Península Ibérica. Procedía principalmente de elefantes africanos y asiáticos, aunque también se documenta marfil de hipopótamo, cachalote e incluso de proboscídeos pleistocenos. Durante el Calcolítico predominó el marfil africano en el suroeste peninsular y el asiático en el sureste, mientras que las rutas orientales perdieron importancia al final de la Edad del Cobre y comienzos de la Edad del Bronce.\n"
     "Yacimientos como Los Millares, Valencina de la Concepción y Cabezo Redondo muestran la relevancia de estos materiales en la fabricación de objetos suntuarios y su asociación con contextos funerarios y grupos de élite, evidenciando la existencia de redes de intercambio suprarregionales y mecanismos de diferenciación social.",
     "1166_Producción y consumo de marfil_2022.pdf; 1043_ambar_2018.pdf; https://historia.nationalgeographic.com.es/a/uso-ambar-peninsula-iberica-paleolitico-hasta-edad-bronce_21014", "compleja", "es", 3),

    (31,
     "¿Qué procesos tafonómicos afectan a la interpretación de los depósitos funerarios prehistóricos en cuevas de la Península Ibérica y cómo se han abordado metodológicamente?",
     "La interpretación de los depósitos funerarios prehistóricos en cuevas requiere considerar los procesos tafonómicos, es decir, el conjunto de transformaciones que afectan a los restos desde la muerte de los individuos hasta su recuperación arqueológica. Estos procesos incluyen tanto fenómenos bioestratinómicos, ocurridos entre la muerte y el enterramiento, como diagenéticos, producidos después del depósito de los restos.\n"
     "Entre los principales procesos tafonómicos documentados destacan la fracturación ósea, cuya morfología permite distinguir fracturas producidas en vida (ante mortem), durante el depósito (peri mortem) o tras el enterramiento (post mortem); las termoalteraciones, identificadas mediante cambios de color, textura y deformación del hueso; las marcas de carnívoros, como mordeduras y arrastres; el pisoteo, que provoca desplazamientos, fracturas y estriaciones; la meteorización, causada por variaciones de temperatura y humedad; la disolución de la superficie ósea por la acción de aguas carbonatadas; la concreción calcítica, que recubre los huesos, y las alteraciones producidas por el crecimiento de raíces.\n"
     "Estos procesos pueden modificar la posición original de los restos y generar palimpsestos arqueológicos, dificultando la reconstrucción de las prácticas funerarias. En contextos como la cueva sepulcral de El Espinoso (Ribadedeva, Asturias), los problemas de conservación, las alteraciones postdeposicionales y la pérdida de materiales han hecho imprescindible el empleo de estudios tafonómicos para interpretar correctamente los depósitos superficiales. Del mismo modo, en yacimientos de la transición Mesolítico-Neolítico del Mediterráneo occidental, el remontaje de fragmentos cerámicos procedentes de diferentes niveles estratigráficos ha permitido identificar mezclas de materiales producidas por perturbaciones posteriores.\n"
     "Metodológicamente, estos estudios combinan el análisis tafonómico con la cuantificación de los conjuntos óseos, mediante indicadores como el Número de Restos (NR), el Número Mínimo de Elementos (NME) y el Número Mínimo de Individuos (NMI), así como índices de conservación como la Tasa de Fragmentación (TF) y el Índice de Representación Anatómica (IRA). Además, integran el análisis espacial de los restos, la información estratigráfica y las dataciones radiocarbónicas mediante modelos estadísticos bayesianos, junto con el remontaje de piezas cerámicas y líticas para identificar desplazamientos entre niveles y reconocer procesos de formación del depósito. Este enfoque multidisciplinar permite diferenciar las prácticas funerarias originales de las alteraciones producidas por agentes biológicos, físicos y antrópicos, mejorando la interpretación de los contextos funerarios prehistóricos.",
     "1008_TafonomiaMarcoMetodologico_2017.pdf; 1061_Taphonomic_2018.pdf; 1124_The human remains_2020.pdf; 1035_Chronological reassessment_2018.pdf", "compleja", "es", 4),
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
