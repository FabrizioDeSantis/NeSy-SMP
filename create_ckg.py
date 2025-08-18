from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD
from utils import visualize
import networkx as nx

CKG = Namespace("http://example.org/clinical-guideline/")
SNOMED = Namespace("http://snomed.info/id/")
SCHEMA = Namespace("https://schema.org/")

g = Graph()

g.bind("snomed", SNOMED)
g.bind("rdf", RDF)
g.bind("rdfs", RDFS)
g.bind("owl", OWL)
g.bind("sct", XSD)
g.bind("sct", SNOMED)
g.bind("schema", SCHEMA)

# Patient
#g.add((SNOMED.Patient, RDF.type, OWL.Class))
g.add((SNOMED.Patient, SNOMED.hasOutcome, SNOMED.Death))
g.add((SNOMED.Patient, SNOMED.hasOutcome, SNOMED.PatientDischargedAlive))

# Diseases
# g.add((SNOMED.Cancer, RDF.type, OWL.Class))
# g.add((SNOMED.Pneumonia, RDF.type, OWL.Class))
# g.add((SNOMED.Sepsis, RDF.type, OWL.Class))

# Outcomes
# g.add((SNOMED.Death, RDF.type, OWL.Class))
g.add((SNOMED.Death, RDFS.subClassOf, SNOMED.Outcome))
g.add((SNOMED.Death, RDF.type, SNOMED.Outcome))
# g.add((SNOMED.PatientDischargedAlive, RDF.type, OWL.Class))
g.add((SNOMED.PatientDischargedAlive, RDFS.subClassOf, SNOMED.Outcome))

# Patient attributes
g.add((CKG.hasAge, RDF.type, OWL.ObjectProperty))
g.add((CKG.hasAge, RDFS.domain, SNOMED.Patient))
g.add((CKG.hasAge, RDFS.range, SNOMED.Age))

# g.add((CKG.hasCRP, RDF.type, OWL.ObjectProperty))
# g.add((CKG.hasCRP, RDFS.domain, SNOMED.Patient))
# g.add((CKG.hasCRP, RDFS.range, SNOMED.CReactiveProtein))

# g.add((CKG.hasLactate, RDF.type, OWL.ObjectProperty))
# g.add((CKG.hasLactate, RDFS.domain, SNOMED.Patient))
# g.add((CKG.hasLactate, RDFS.range, SNOMED.Lactate))

g.add((CKG.hasBMI, RDF.type, OWL.ObjectProperty))
g.add((CKG.hasBMI, RDFS.domain, SNOMED.Patient))
g.add((CKG.hasBMI, RDFS.range, SNOMED.BodyMassIndex))

# g.add((SNOMED.Patient, SNOMED.has, SNOMED.Age))
# g.add((SNOMED.Patient, SNOMED.has, SNOMED.CReactiveProtein))
# g.add((SNOMED.Patient, SNOMED.has, SNOMED.Lactate))
# g.add((SNOMED.Patient, SNOMED.has, SNOMED.BodyMassIndex))

# Comorbidities
# g.add((SNOMED.Patient, SNOMED.canHave, SNOMED.Cancer))
# g.add((SNOMED.Patient, SNOMED.canHave, SNOMED.Pneumonia))
# g.add((SNOMED.Patient, SNOMED.canHave, SNOMED.HIV))

# RiskFactor for death
# g.add((SCHEMA.riskFactor, RDF.type, OWL.Class))
g.add((SNOMED.AgeRiskFactor, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.AgeRiskFactor, SCHEMA.propertyId, SNOMED.Age))
g.add((SNOMED.AgeRiskFactor, SNOMED.hasOutcome, SNOMED.Death))
g.add((SNOMED.AgeRiskFactor, SCHEMA.greaterOrEqual, Literal(65, datatype=XSD.integer)))
g.add((SNOMED.AgeRiskFactor, SCHEMA.increaseRiskOf, SNOMED.Sepsis))

g.add((SNOMED.CReactiveProteinRiskFactor, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.CReactiveProteinRiskFactor, SCHEMA.propertyId, SNOMED.CReactiveProtein))
g.add((SNOMED.CReactiveProteinRiskFactor, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((SNOMED.CReactiveProteinRiskFactor, SCHEMA.greaterOrEqual, Literal(100, datatype=XSD.integer)))

g.add((SNOMED.LactateRiskFactor, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.LactateRiskFactor, SCHEMA.propertyId, SNOMED.Lactate))
g.add((SNOMED.LactateRiskFactor, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((SNOMED.LactateRiskFactor, SCHEMA.greaterOrEqual, Literal(4, datatype=XSD.integer)))

g.add((SNOMED.BodyMassIndexRiskFactor, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.BodyMassIndexRiskFactor, SCHEMA.propertyId, SNOMED.BodyMassIndex))
g.add((SNOMED.BodyMassIndexRiskFactor, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((SNOMED.BodyMassIndexRiskFactor, SCHEMA.greaterOrEqual, Literal(24, datatype=XSD.integer)))

g.add((SNOMED.WhiteBloodCellsRiskFactor, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.WhiteBloodCellsRiskFactor, SCHEMA.propertyId, SNOMED.WhiteBloodCells))
g.add((SNOMED.WhiteBloodCellsRiskFactor, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((SNOMED.WhiteBloodCellsRiskFactor, SCHEMA.greaterOrEqual, Literal(30, datatype=XSD.integer)))

g.add((SNOMED.BilirubinRiskFactor, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.BilirubinRiskFactor, SCHEMA.propertyId, SNOMED.Bilirubin))
g.add((SNOMED.BilirubinRiskFactor, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((SNOMED.BilirubinRiskFactor, SCHEMA.greaterOrEqual, Literal(2, datatype=XSD.float)))

g.add((SNOMED.PlateletRiskFactor, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.PlateletRiskFactor, SCHEMA.propertyId, SNOMED.Platelet))
g.add((SNOMED.PlateletRiskFactor, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((SNOMED.PlateletRiskFactor, SCHEMA.lessOrEqual, Literal(50, datatype=XSD.integer)))

g.add((SNOMED.MeanArterialPressureRiskFactor, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.MeanArterialPressureRiskFactor, SCHEMA.propertyId, SNOMED.MeanArterialPressure))
g.add((SNOMED.MeanArterialPressureRiskFactor, SCHEMA.increaseRiskOf, SNOMED.Death))
g.add((SNOMED.MeanArterialPressureRiskFactor, SCHEMA.lessOrEqual, Literal(65, datatype=XSD.integer)))

g.add((SNOMED.Cancer, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.Cancer, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.Cancer, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.Pneumonia, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.Pneumonia, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.Pneumonia, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.HIV, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.HIV, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.HIV, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.CirrhosisOfLiver, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.CirrhosisOfLiver, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.CirrhosisOfLiver, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.KidneyDisease, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.KidneyDisease, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.KidneyDisease, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.CardiacInsufficiency, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.CardiacInsufficiency, RDFS.subClassOf, SNOMED.Comorbidity))
g.add((SNOMED.CardiacInsufficiency, SCHEMA.increaseRiskOf, SNOMED.Death))

g.add((SNOMED.SepticShock, RDFS.subClassOf, SCHEMA.riskFactor))
g.add((SNOMED.SepticShock, SCHEMA.increaseRiskOf, SNOMED.Death))

# RiskFactor for sepsis
g.add((SNOMED.Cancer, SCHEMA.increasesRiskOf, SNOMED.Sepsis))
g.add((SNOMED.Pneumonia, SCHEMA.increasesRiskOf, SNOMED.Sepsis))
g.add((SNOMED.HIV, SCHEMA.increasesRiskOf, SNOMED.Sepsis))
g.add((SNOMED.CirrhosisOfLiver, SCHEMA.increasesRiskOf, SNOMED.Sepsis))
g.add((SNOMED.KidneyDisease, SCHEMA.increasesRiskOf, SNOMED.Sepsis))
g.add((SNOMED.CardiacInsufficiency, SCHEMA.increasesRiskOf, SNOMED.Sepsis))
g.add((SNOMED.SepticShock, SCHEMA.increasesRiskOf, SNOMED.Sepsis))

visualize(g)