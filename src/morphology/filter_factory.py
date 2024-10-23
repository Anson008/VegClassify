from src.morphology import criteria


class FilterFactory:
    def get_filter(self, field, operator, threshold):
        if operator == ">=":
            return criteria.GreaterThanOrEqualToCriteria(field, threshold)
        if operator == ">":
            return criteria.GreaterThanCriteria(field, threshold)
        if operator == "<=":
            return criteria.LessThanOrEqualToCriteria(field, threshold)
        if operator == "<":
            return criteria.LessThanCriteria(field, threshold)