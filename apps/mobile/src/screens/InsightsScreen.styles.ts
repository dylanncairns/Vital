import { StyleSheet } from "react-native";

export const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: "#F7F8FB",
  },
  container: {
    paddingHorizontal: 16,
    paddingBottom: 28,
    gap: 12,
  },
  header: {
    paddingTop: 14,
    paddingBottom: 10,
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
    minHeight: 44,
  },
  headerTitle: {
    fontSize: 28,
    fontFamily: "Exo2-Bold",
    color: "#101426",
    textAlign: "center",
  },
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: "#E6E9F2",
    gap: 2,
  },
  title: {
    fontSize: 16,
    fontFamily: "Exo2-Bold",
    color: "#101426",
  },
  summary: {
    fontSize: 14,
    fontFamily: "Exo2-Regular",
    color: "#343A52",
  },
  meta: {
    fontSize: 12,
    fontFamily: "Exo2-Regular",
    color: "#69708A",
  },
  badge: {
    alignSelf: "flex-start",
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  badgeText: {
    fontSize: 12,
    fontFamily: "Exo2-SemiBold",
  },
  citationsHeader: {
    marginTop: 8,
    fontSize: 13,
    fontFamily: "Exo2-Bold",
    color: "#232A44",
  },
  citationLine: {
    fontSize: 12,
    fontFamily: "Exo2-Regular",
    color: "#343A52",
  },
  emptyText: {
    marginTop: 24,
    textAlign: "center",
    color: "#5D6480",
    fontSize: 14,
    fontFamily: "Exo2-Regular",
  },
});
