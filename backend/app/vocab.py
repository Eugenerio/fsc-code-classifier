"""Query-side thesaurus: everyday product words -> the terse words the FSC catalog uses for the same things.

FSC descriptions say "Footwear, Men", "Hand Tools, Non-edged" and "Engine Electrical Systems Components"; a company
site says "combat boots", "wrenches" and "alternators". BM25 cannot bridge that gap, so each line below lists words a
company might use (left of `=>`, separated by `|`, one to three words each) and the catalog words that name their
class (right). Expansions are appended to a query's tokens, never to the indexed descriptions, and the original words
stay in the query. This is a no-LLM fallback: with Gemini the embedding and LLM channels cover the same ground.

Keep keys unambiguous ("plant", "range" or "line" would fire on any site) and expansions made of catalog words.
"""

from __future__ import annotations

BRIDGE = """
# --- clothing, textiles, personal equipment (groups 83, 84) ---
uniform | jacket | coat | parka | shirt | pants | trousers | coverall | apron | sweater | hoodie | raincoat => clothing outerwear special purpose
boot | boots | shoe | sneaker | sandal | footwear => footwear
glove | mitten | sock | hosiery | handwear => handwear hosiery clothing
hat | helmet | beanie | headgear => clothing special purpose individual equipment
backpack | rucksack | duffel | luggage | suitcase => individual equipment luggage
tactical vest | plate carrier | load bearing | webbing gear | tactical gear => individual equipment clothing special purpose
body armor | armor plate | ballistic | bulletproof | kevlar => armor personal
ppe | protective clothing | flame resistant | work wear | workwear => clothing special purpose individual equipment
tent | tarp | tarpaulin | canopy => tent tarpaulin
flag | banner | pennant => flag pennant
patch | insignia | emblem | badge | name tape => badge insignia
fabric | cloth | textile | cotton | nylon | canvas => textile fabric
thread | yarn | sewing => yarn thread
zipper | buckle | velcro | button => notion apparel finding
leather | hide => leather
# --- medical, dental, veterinary (group 65) ---
surgical glove | exam glove | examination glove | nitrile glove | latex glove | medical glove | syringe | needle | scalpel | forceps | suture | catheter | stethoscope | thermometer clinical | surgical instrument => medical surgical instrument supply
bandage | gauze | dressing | wound care => surgical dressing material
wheelchair | stretcher | hospital bed | gurney | crutch | patient bed | patient monitor | patient monitoring => hospital furniture utensil supply
pharmaceutical | medication | medicine | vaccine | antibiotic | pharmacy => drug biological
dental | dentist | orthodontic => dental instrument supply
x-ray | radiology | mri | ultrasound => x-ray supply medical dental veterinary
scrubs | surgical gown | surgical mask | isolation gown | face mask => hospital surgical clothing special purpose
eyeglass | spectacle | ophthalmic => ophthalmic instrument supply
diagnostic | reagent | test kit | assay => in-vitro diagnostic substance reagent test kit
first aid | med kit | trauma kit => replenishable field medical set kit outfit
# --- computers, office (groups 70, 74, 75, 71) ---
!laptop | notebook computer | !desktop | workstation | personal computer | !server | mainframe | tablet | thin client => computer central processing unit digital
monitor display | !printer | scanner | keyboard | hard drive | solid state drive | flash drive | tape drive | peripheral => input output storage device
router | network switch | firewall | modem | networking | ethernet | wifi => communication equipment miscellaneous component
software | operating system => software
toner | ink cartridge | printer ribbon => supply
desk | office chair | bookcase | credenza | conference table | office furniture => office furniture
filing cabinet | locker | shelving | storage cabinet | storage rack => cabinet locker bin shelving
sofa | couch | mattress | dresser | household furniture => household furniture
pen | pencil | marker | stapler | folder | binder | envelope | paper clip | highlighter | office supply => office supply device accessory
copy paper | stationery | letterhead | notepad | record form => stationery record form
calculator | typewriter | shredder | copier | photocopier | fax => office machine accounting calculating
# --- cleaning, toiletries, packaging (groups 79, 85, 81) ---
vacuum | floor scrubber | scrubber | floor polisher | carpet cleaner | floor machine | burnisher => floor polisher vacuum cleaning equipment
mop | broom | brush | sponge | squeegee | duster | janitorial => broom brush mop sponge
detergent | cleaner | degreaser | sanitizer | bleach | cleaning chemical | janitorial chemical | floor polish | cleaning solution => cleaning polishing compound preparation
trash bag | garbage bag | bin liner | sack | burlap => bag sack
paper towel | toilet paper | toilet tissue | napkin | facial tissue => toiletry paper product
hand soap | shampoo | toothpaste | deodorant | razor | toothbrush | lotion | soap => toiletry personal article soap shave dentifrice
box | carton | crate | cardboard box | corrugated => box carton crate
drum | barrel | pail | bucket | tin can | steel drum => drum can
bottle | jar | vial => bottle jar
pallet | tote | shipping container | intermodal | conex => special ship storage container
shrink wrap | bubble wrap | stretch wrap | strapping | packing tape | packing foam | packaging material | cushioning => packaging packing bulk material
gas cylinder | propane tank | compressed gas cylinder | cylinder rental => commercial industrial gas cylinder
reel | spool | bobbin | cable reel => reel spool
# --- fasteners, hardware, abrasives (group 53) ---
screw | machine screw | lag screw | set screw | cap screw | self-tapping | sheet metal screw => screw
bolt | hex bolt | carriage bolt | anchor bolt | u-bolt | eye bolt | hex head => bolt
nut | lock nut | wing nut | washer | lock washer | flat washer => nut washer
rivet | pop rivet | blind rivet => rivet
cotter pin | dowel | clevis pin | roll pin | nail | staple | tack => nail machine key pin
clamp | retaining ring | snap ring | hook and loop | zip tie | cable tie | fastener | anchor => fastening device
gasket | oil seal | shaft seal | gland packing | sealing ring | seal kit => packing gasket material
spring | compression spring | torsion spring | coil spring | leaf spring | wire form => coil flat wire spring
bushing | shim | spacer | bearing sleeve | grommet => bushing ring shim spacer
hinge | latch | hasp | door handle | knob | padlock | caster | door hardware | cabinet hardware | turnbuckle | hardware => hardware commercial
bracket | mounting bracket | angle bracket => bracket
stud | threaded rod | all thread => stud
abrasive | sandpaper | grinding wheel | sanding disc | cutoff wheel | flap disc | emery => disk stone abrasive material
# --- hand tools, measuring, machine tools (groups 51, 52, 34, 35) ---
wrench | pliers | screwdriver | socket set | ratchet | !hammer | mallet | pry bar | crowbar | vise | allen key | hex key => hand tool non-edged nonpowered
chisel | knife | !saw | hacksaw | rasp | !shears | scissors | utility knife | axe | hatchet | snips => hand tool edged nonpowered
!drill | cordless drill | impact driver | !grinder | sander | nail gun | angle grinder | power tool | jigsaw | circular saw | heat gun => hand tool power driven
drill bit | twist drill | reamer | countersink | counterbore | hole saw => drill bit counterbore countersink
tap and die | threading | collet => tap die collet hand machine
toolbox | tool chest | tool bag | tool cabinet => tool hardware box
tool kit | tool set | mechanic tool set => specialized kit outfit hand tool
caliper | micrometer | gauge | gage | height gauge | thread gauge | go no-go | bore gauge | plug gauge | dial indicator => inspection gage precision layout tool
tape measure | ruler | combination square | spirit level | protractor | measuring tape | scribe | straightedge => measuring tool craftsmen
cutting tool | end mill | endmill | carbide insert | milling cutter | lathe tool | broach | hob | !cutter => cutting tool machinery tool
cnc lathe | turning center | lathe => lathe
machining center | milling machine | vertical mill | cnc mill => machining center way-type milling machine
surface grinder | cylindrical grinder | grinding machine => grinding machine
press brake | brake press | punch press | stamping press | hydraulic press | forging press => press power driven manual
plasma cutter | laser cutter | waterjet | punching machine | shearing machine => punching shearing machinery
band saw | bandsaw | cold saw | cutoff saw => saw filing machine
welding machine | welder | mig | tig | arc welder | stick welder | spot welder | spot welding => welding equipment elect arc resistance
welding wire | welding rod | welding electrode | flux | brazing | soldering | solder => welding soldering brazing supply accessory
oxy-acetylene | oxyacetylene | cutting torch | acetylene torch | torch => gas welding heat cutting metalizing equipment
jig | tooling | template | production fixture => production jig fixture template
# --- aerospace (groups 15, 16, 17, 49) ---
airframe | fuselage | wing | bulkhead | stringer | spar | fairing | nacelle | skin panel | sheetmetal aircraft | sheet metal aircraft | aircraft structure | aircraft structural => airframe structural component
landing gear | aircraft wheel | aircraft brake => aircraft landing gear wheel brake component
propeller | rotor blade | helicopter => aircraft propeller helicopter rotor blade drive mechanism
avionics | cockpit | altimeter | gyro | autopilot | transponder | attitude indicator | flight instrument => flight instrument navigation aircraft
hydraulic actuator | servo valve | hydraulic cylinder | accumulator | hydraulic power => aircraft hydraulic vacuum system component
mil-spec hardware | aerospace hardware | aircraft hardware | aircraft fastener => aircraft component hardware weapon system
parachute | cargo net | tie down | tiedown | cargo strap | ratchet strap | airdrop => parachute aerial delivery recovery system cargo tie down
ground support equipment | aircraft tug | tow bar | aircraft jack | aircraft stand => aircraft ground servicing equipment maintenance repair shop specialized
# --- engines, power transmission, electrical machines (groups 28, 29, 30, 31, 61) ---
diesel engine | diesel => diesel engine component
turbine | jet engine | turbofan | turboprop => gas turbine jet engine component
alternator | starter motor | ignition | spark plug | glow plug | ignition coil => engine electrical system component
radiator | water pump | thermostat | coolant | cooling fan => engine cooling system component
fuel injector | carburetor | fuel filter | fuel line | fuel tank => engine fuel system component
air filter | oil filter | filter element | strainer => engine air oil filter strainer cleaner
turbocharger | supercharger => turbosupercharger
gearbox | transmission | speed reducer | clutch | drivetrain | differential => torque converter speed changer power transmission
sheave | idler pulley | v-pulley => gear pulley sprocket transmission chain
belt | v-belt | serpentine belt | conveyor belt | fan belt | timing belt => belting drive belt fan belt accessory
shaft coupling | universal joint | u-joint => power transmission equipment miscellaneous
ball bearing | roller bearing | needle bearing | thrust bearing => bearing anti-friction unmounted
pillow block | flange bearing | mounted bearing => bearing mounted
plain bearing | sleeve bearing | journal bearing => bearing plain unmounted
electric motor | servo motor | gear motor | ac motor | dc motor | stepper motor => motor electric
generator | genset | standby generator | portable generator => generator generator set electric
transformer | power supply | inverter | ups | rectifier | converter | voltage regulator | battery charger => transformer converter electric rotating nonrotating
battery | lithium battery | lead acid | forklift battery | rechargeable | alkaline | battery pack => battery rechargeable nonrechargeable
solar panel | photovoltaic | solar => solar electric power system
# --- electrical and electronic components (groups 58, 59, 60, 62, 63, 66, 67) ---
wire harness | harness | jumper cable | patch cord | power cord | extension cord | cord set => cable cord wire assembly
connector | terminal | lug | crimp | splice | terminal block | receptacle | coax => connector electric lug terminal strip
resistor | capacitor | inductor | diode | transistor | integrated circuit | microcontroller | semiconductor | microchip | fpga => microcircuit semiconductor device resistor capacitor
circuit board | pcb | printed circuit | board assembly | electronic assembly => electric electronic assembly board card associated hardware
relay | solenoid | contactor => relay solenoid
toggle switch | pushbutton | rocker switch | limit switch | pressure switch => switch
fuse | circuit breaker | breaker | surge protector => fuse arrestor absorber protector circuit breaker
sensor | transducer | thermocouple | proximity sensor | load cell => pressure temperature humidity measurement control instrument
antenna | two-way radio | transceiver | walkie talkie | satellite => radio communication equipment antenna navigation
radar | sonar | lidar => radar equipment
headset | microphone | speaker | earpiece | headphones | intercom => headset handset microphone speaker
camera | cctv | surveillance | security camera | camcorder => camera still picture motion picture
alarm | smoke detector | fire alarm | intrusion | security system | access control | burglar alarm | siren => alarm signal security detection system
led light | lighting | lamp | bulb | luminaire | light fixture | floodlight | street light | fluorescent | halogen => electric lighting fixture lamp
flashlight | headlamp | lantern | work light | spotlight => electric portable hand lighting
ballast | lampholder => ballast lampholder starter
night vision | thermal imager | infrared camera => night vision emitted reflected radiation
fiber optic | optical fiber | optical cable => fiber optic
# --- pumps, valves, pipe, plumbing, HVAC, safety (groups 41-49, 42) ---
pump | centrifugal pump | submersible pump | hydraulic pump | diaphragm pump | gear pump | sump pump | vacuum pump => pump compressor
air compressor | gas compressor | compressor => compressor vacuum pump
valve | gate valve | ball valve | check valve | butterfly valve | globe valve | solenoid valve | relief valve | control valve | needle valve | plug valve | actuated valve | regulator => valve powered nonpowered
hose | hydraulic hose | air hose | flexible hose | rubber hose | braided hose => hose tubing flexible
elbow | tee fitting | flange | pipe union | nipple | adapter | quick disconnect | hydraulic fitting | tube fitting | pipe fitting => fitting specialty hose pipe tube
filtration | separator | strainer | filter housing | cartridge filter | coalescer => centrifugal separator pressure vacuum filter
faucet | sink | toilet | urinal | shower | plumbing | lavatory | drain | plumbing fixture => plumbing fixture accessory
water heater | boiler | furnace | space heater | burner | heating => space heat equipment domestic water heater fuel burning
air conditioner | hvac | chiller | refrigerator | freezer | refrigeration | condenser | evaporator => refrigeration air conditioning equipment component
ventilation | exhaust fan | air handler | ventilator | blower => fan air circulator blower
water treatment | water filter | water purifier | reverse osmosis | desalination | water softener => water purification equipment
sewage | wastewater | septic | effluent => sewage treatment equipment
fire extinguisher | extinguisher | fire suppression | sprinkler | fire hose | firefighting => fire fighting equipment
respirator | gas mask | safety harness | fall protection | hard hat | safety glasses | goggles | hearing protection | high visibility | eyewash | safety shower | life jacket | lifejacket | lanyard => safety rescue equipment
spill kit | oil spill | containment boom | hazmat | absorbent pad => hazardous material spill containment cleanup material
# --- construction, building, metals (groups 54, 55, 56, 93, 95) ---
cement | concrete | aggregate | gravel | crushed stone | mortar | asphalt => mineral construction material bulk
brick | masonry | ceramic tile | paver => building tile brick block
drywall | gypsum | insulation | sheetrock | vapor barrier => wallboard building paper thermal insulation material
roofing | shingle | siding | gutter | flashing => roofing siding material
fence | fencing | chain link | barbed wire | guardrail => fencing fence gate component
door | window | skylight | garage door | modular building => building component prefabricated
lumber | timber | plywood | osb | millwork | molding | hardwood | softwood => lumber related basic wood material millwork plywood veneer
rebar | steel beam | i-beam | angle iron | structural steel => structural shape
steel sheet | aluminum sheet | metal sheet | steel coil | shim stock | stainless sheet | steel plate | aluminum plate => plate sheet strip foil
steel bar | round bar | flat bar | bar stock | aluminum bar | brass rod | copper rod | billet => bar rod
steel wire | galvanized wire | spring wire | baling wire => wire nonelectrical
scaffolding | formwork | concrete form | shoring | ladder => scaffold equipment concrete form
storage tank | water tank | silo | pressure vessel => storage tank
portable building | shed | kiosk | quonset | trailer office => prefabricated portable building
tower | mast | utility pole | light pole => prefabricated tower structure
rubber sheet | rubber mat | neoprene | silicone rubber | elastomer => rubber fabricated material
plastic sheet | acrylic | polycarbonate | pvc | hdpe | ptfe | teflon | polyethylene | delrin | fiberglass => plastic fabricated material
glass sheet | tempered glass | window glass | mirror => glass fabricated material
# --- chemicals, paints, fuels, lubricants (groups 68, 80, 91) ---
solvent | acetone | thinner | caustic | resin | industrial chemical | lab chemical => chemical
dye | pigment | colorant => dye
oxygen | nitrogen | argon | helium | carbon dioxide | hydrogen | compressed gas | liquefied gas | propane | refrigerant | acetylene => gas compressed liquefied
pesticide | herbicide | insecticide | rodenticide | disinfectant | fungicide | pest control | weed killer => pest control agent disinfectant
paint | primer | coating | lacquer | varnish | enamel | topcoat | powder coat | epoxy coating => paint dope varnish
paintbrush | paint roller | artist brush => paint artist brush
sealant | caulk | silicone sealant | sealer | putty | rust inhibitor | corrosion inhibitor | anti-seize | thread locker | threadlocker | waterproofing => preservative sealing compound
adhesive | glue | epoxy | bonding agent | cyanoacrylate | contact cement | hot melt => adhesive
diesel fuel | gasoline | jet fuel | kerosene | jp-8 | avgas | biodiesel => liquid propulsion fuel petroleum base
heating oil | fuel oil | bunker fuel => fuel oil
lubricant | grease | motor oil | engine oil | hydraulic oil | gear oil | cutting oil | lubricating oil | synthetic oil | penetrating oil | transmission fluid | hydraulic fluid | antifreeze => oil grease cutting lubricating hydraulic
paraffin | beeswax | tallow => miscellaneous wax oil fat
coal | charcoal | coke | firewood | briquette => solid fuel
# --- food, animals, agricultural supplies (groups 87, 88, 89) ---
meat | beef | pork | lamb | chicken | turkey | seafood | shrimp | sausage | bacon | frozen meat => meat poultry fish
milk | cheese | butter | yogurt | egg | dairy | ice cream => dairy food egg
vegetable | fruit | fresh produce | potato | onion | tomato | salad => fruit vegetable
bread | bakery | cake | cookie | pastry | flour | rice | pasta | noodle | grain | oats | cracker => bakery cereal product
candy | chocolate | peanut | confection | snack | syrup | honey | sweetener => sugar confectionery nut
jam | jelly | marmalade | fruit spread => jam jelly preserve
soup | broth | bouillon => soup bouillon
infant formula | baby food | dietary supplement | protein powder | vitamin | special diet | gluten free => special dietary food specialty preparation
cooking oil | vegetable oil | olive oil | shortening | margarine => food oil fat
condiment | spice | sauce | ketchup | mustard | mayonnaise | vinegar | seasoning => condiment related product
coffee | tea | cocoa | espresso => coffee tea cocoa
bottled water | soda | juice | soft drink | energy drink | lemonade | sparkling water | drinking water | beverage => beverage nonalcoholic
beer | wine | liquor | whiskey | vodka | alcoholic beverage => beverage alcoholic
mre | meals ready | field ration | combat ration | ration => composite food package
cigarette | cigar | tobacco | snuff => tobacco product
cattle | livestock | hog | sheep | goat | live animal => live animal food
working dog | k9 | laboratory animal | research animal => live animal not food
hay | animal feed | fodder | forage | pet food | livestock feed | alfalfa => forage feed
fertilizer | compost | manure | potash | soil amendment => fertilizer
seed | nursery stock | sapling | sod | seedling => seed nursery stock
# --- vehicles, marine, rail, materials handling (groups 20, 22, 23, 24, 25, 26, 38, 39, 40) ---
car | sedan | suv | passenger vehicle | minivan | automobile => passenger motor vehicle
truck | tractor trailer | dump truck | pickup truck | box truck | tow truck | utility truck => truck tractor wheeled
trailer | flatbed trailer | utility trailer | cargo trailer | lowboy => trailer
motorcycle | scooter | bicycle | atv | moped => motorcycle motor scooter bicycle
tractor | farm tractor | bulldozer | dozer | crawler => tractor full track wheeled
brake pad | brake shoe | brake rotor | axle | wheel hub | steering | tie rod | drive shaft | suspension | shock absorber | strut | cv joint | wheel bearing | lug nut => vehicle brake steer axle wheel track component
bumper | fender | hood | chassis | truck bed | tailgate | running board | grille | hitch | body panel => vehicle cab body frame structural component
seat cover | floor mat | dashboard | windshield | wiper | wiper blade | horn | bed liner | mud flap | cargo rack => vehicle furniture accessory miscellaneous vehicle component
exhaust | muffler | catalytic | tailpipe | exhaust pipe => miscellaneous vehicle component engine accessory
tire | tyre | inner tube | retread | aircraft tire => tire tube
boat | ship | yacht | outboard | marine engine | life raft | marine hardware | hull => ship boat marine equipment hardware
fishing net | fishing gear | fishing tackle | fishing line | lobster trap => commercial fishing equipment
railroad | locomotive | railcar | rail tie | railroad spike | rail car => railway equipment locomotive rail car track material
wire rope | aircraft cable | steel cable | galvanized cable | stainless cable | cable sling | wire sling | control cable | swaged cable => chain wire rope
lifting chain | anchor chain | tow chain | load chain | chain sling | roller chain => chain wire rope
nylon rope | polyester rope | manila rope | cordage | twine | paracord | bungee => fiber rope cordage twine
thimble | wire rope clip | swage | swaging | ferrule | shackle | rope fitting | end fitting | cable fitting | cable terminal => fitting rope cable chain
sling | hoist | winch | tackle | rigging | chain hoist | come along | snatch block | lifting equipment => block tackle rigging sling winch hoist crane derrick
crane | derrick | overhead crane | jib crane | gantry => winch hoist crane derrick
forklift | pallet jack | hand truck | dolly | lift truck | reach truck => warehouse truck tractor self-propelled materials handling nonself-propelled
conveyor | roller conveyor | belt conveyor | hopper => conveyor materials feeder
elevator | escalator | dumbwaiter => elevator escalator
# --- oil and gas, mining, construction machinery (groups 38, 49, 43) ---
drill pipe | drill collar | drill string | drilling rig | mud motor | downhole motor | roller cone | pdc bit | tricone | drilling tool | rock bit | rock drill | auger | horizontal directional drilling | tunnel boring | drill steel => mining rock drilling earth boring related
fishing tool | overshot | slickline | wireline | coiled tubing | !packer | bridge plug | fracturing | perforating | completion tool | liner hanger | cementing | tubing hanger | centralizer | octg | oil country tubular => petroleum production distribution
wellhead | christmas tree | blowout preventer | bop | pressure control | choke manifold | pumpjack | pump jack | sucker rod | artificial lift | oil field | well service => petroleum production distribution
pipeline | pig launcher | tank farm | fuel dispensing | refueling | fuel nozzle | fueling => petroleum distribution lubrication fuel dispensing
excavator | backhoe | loader | grader | skid steer | earthmoving | trencher | compactor => earth moving excavating
snow plow | street sweeper | road sweeper | salt spreader | snow blower => road clearing cleaning
# --- instruments, lab, optics, misc (groups 66, 67, 69, 73, 76-78, 99) ---
pressure gauge | temperature gauge | flow meter | flowmeter | level sensor | thermostat | hygrometer | manometer | data logger | pyrometer => pressure temperature humidity measurement control instrument
rotameter | water meter | gas meter | liquid level => liquid gas flow level measuring instrument
scale | weighing | truck scale | lab balance | weigh scale => scale balance
microscope | telescope | binocular | spectroscope | rangefinder | riflescope => optical instrument test equipment component accessory
laboratory | lab equipment | beaker | pipette | fume hood | autoclave | incubator | glassware | lab supply | test tube | flask => lab equipment supply
oscilloscope | multimeter | signal generator | spectrum analyzer | voltmeter | ammeter | megger | calibrator | calibration => electric electronic property measuring test instrument
compass | sextant | gyrocompass | direction finder => navigational instrument
clock | watch | timer | stopwatch | chronometer => time measuring instrument
anemometer | barometer | weather station | rain gauge | meteorological => meteorological instrument apparatus
theodolite | total station | surveying | drafting | drafting table => drafting surveying mapping instrument
tensile tester | hardness tester | material testing | nondestructive | ultrasonic testing | borescope | eddy current | dye penetrant => physical property testing inspection equipment
environmental chamber | test chamber | humidity chamber | climate chamber | vibration table => environmental chamber
gas detector | radiation detector | geiger | metal detector | explosive detector | leak detector | air quality => hazard detecting instrument apparatus
photography | tripod | darkroom | photographic | projector => photographic equipment supply
musical instrument | guitar | piano | trumpet | violin | saxophone | sheet music => musical instrument part accessory
sporting goods | athletic | exercise equipment | treadmill | dumbbell | barbell | basketball | soccer | playground | toy | trampoline => athletic sporting recreation gym equipment game toy
cookware | oven | stove | grill | fryer | griddle | dishwasher | commercial kitchen | food service | steamer | microwave | coffee maker | buffet => food cooking baking servicing equipment kitchen appliance
utensil | flatware | cutlery | dinnerware | silverware | tableware | mug | tray => cutlery flatware tableware kitchen hand tool utensil
printing press | bookbinding | duplicator | label printer | engraving | embossing | laser engraver | industrial marking => print duplicating bookbinding equipment industrial marking
book | magazine | newspaper | periodical | pamphlet | textbook | atlas | globe => book pamphlet newspaper periodical map chart globe
signage | nameplate | placard | decal | trophy | plaque | memorial | jewelry | collectible | souvenir | casket | funeral => sign advertising display plate memorial jewelry
simulator | training aid | training device | flight simulator => training aid device
"""
